# Adapted from MetaClaw
"""
SkillClaw CLI entry point.

Usage:
    skillclaw setup          — interactive first-time configuration wizard
    skillclaw start          — start the proxy + skill injection
    skillclaw stop           — stop a running SkillClaw instance
    skillclaw status         — check whether SkillClaw is running
    skillclaw config KEY VAL — set a config value
    skillclaw config show    — show current config
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import click
except ImportError:
    print("SkillClaw requires 'click'. Install it with: pip install click")
    sys.exit(1)

from . import runtime_state
from .config_store import CONFIG_FILE, ConfigStore


def _default_daemon_log_path() -> Path:
    return Path.home() / ".skillclaw" / "skillclaw.log"


def _effective_proxy_port(config_store: ConfigStore, override_port: int | None) -> int:
    if override_port:
        return override_port
    return int(config_store.get("proxy.port") or 30000)


def _is_process_alive(pid: int) -> bool:
    return runtime_state.process_alive(pid)


def _read_pid() -> int | None:
    return runtime_state.read_pid()


def _clear_pid():
    runtime_state.clear_pid()


def _clear_pid_if_matches(pid: int):
    runtime_state.clear_pid_if_matches(pid)


def _echo_report(report: dict[str, object]) -> None:
    ordered_keys = [
        "status",
        "integration_scope",
        "config_path",
        "config_exists",
        "expected_model",
        "configured_model",
        "expected_base_url",
        "configured_base_url",
        "configured_provider",
        "proxy_match",
        "expected_skills_dir",
        "skills_dir_exists",
        "skills_dir_mode",
        "legacy_skillclaw_skills_dir",
        "legacy_skillclaw_skills_present",
        "latest_backup",
        "session_boundary_mode",
    ]
    list_keys = {"issues", "notes", "next_steps"}
    emitted: set[str] = set()

    for key in ordered_keys:
        if key not in report:
            continue
        click.echo(f"{key}: {report[key]}")
        emitted.add(key)

    for key, value in report.items():
        if key in emitted or key in list_keys:
            continue
        click.echo(f"{key}: {value}")

    for key in ("issues", "notes", "next_steps"):
        value = report.get(key)
        if not isinstance(value, list):
            continue
        click.echo(f"{key}:")
        if not value:
            click.echo("  (none)")
            continue
        for item in value:
            click.echo(f"  - {item}")


def _healthz_ready(port: int, timeout: float = 0.5) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=timeout) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("ok") is True
    except Exception:
        return False


def _ensure_daemon_not_running():
    pid = _read_pid()
    if pid is None:
        return

    if not _is_process_alive(pid):
        _clear_pid()
        return

    raise click.ClickException(
        f"SkillClaw is already running (PID={pid}). "
        "Use 'skillclaw status' to inspect it or 'skillclaw stop' before starting a new daemon."
    )


def _wait_for_daemon_ready(proc, port: int, log_path: Path, timeout_s: float = 15.0):
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        returncode = proc.poll()
        if returncode is not None:
            raise click.ClickException(f"SkillClaw daemon exited with code {returncode}. Check logs: {log_path}")
        if _healthz_ready(port):
            return
        time.sleep(0.2)

    raise click.ClickException(f"SkillClaw daemon did not become healthy in time. Check logs: {log_path}")


def _daemon_ready_timeout_seconds(default: float = 15.0) -> float:
    raw = str(os.environ.get("SKILLCLAW_DAEMON_READY_TIMEOUT_S", "")).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _spawn_daemon_process(
    port: int | None,
    log_file: str | None,
    effective_port: int,
) -> tuple[int, Path]:
    import os
    import signal
    import subprocess

    log_path = Path(log_file).expanduser() if log_file else _default_daemon_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with runtime_state.daemon_start_lock():
            _ensure_daemon_not_running()

            cmd = [sys.executable, "-m", "skillclaw", "start"]
            if port:
                cmd.extend(["--port", str(port)])

            with log_path.open("ab") as log_handle:
                child_env = os.environ.copy()
                child_env["SKILLCLAW_RUNTIME_KIND"] = "daemon"
                child_env["SKILLCLAW_RUNTIME_LOG_PATH"] = str(log_path)
                popen_kwargs = {
                    "stdin": subprocess.DEVNULL,
                    "stdout": log_handle,
                    "stderr": subprocess.STDOUT,
                    "close_fds": True,
                    "env": child_env,
                }
                if os.name == "nt":
                    creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                    )
                    if creationflags:
                        popen_kwargs["creationflags"] = creationflags
                else:
                    popen_kwargs["start_new_session"] = True
                proc = subprocess.Popen(cmd, **popen_kwargs)

            try:
                _wait_for_daemon_ready(
                    proc,
                    effective_port,
                    log_path,
                    timeout_s=_daemon_ready_timeout_seconds(),
                )
            except Exception:
                try:
                    if proc.poll() is None:
                        if os.name == "nt":
                            proc.terminate()
                        else:
                            os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            if os.name == "nt":
                                proc.kill()
                            else:
                                os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait(timeout=5)
                except Exception:
                    pass

                _clear_pid_if_matches(proc.pid)
                raise

            return proc.pid, log_path
    except RuntimeError as exc:
        owner_pid = exc.args[0] if exc.args else "?"
        raise click.ClickException(
            f"Another 'skillclaw start --daemon' is already in progress (PID={owner_pid}). "
            "Wait for it to finish or stop that process before retrying."
        ) from None


@click.group()
def skillclaw():
    """SkillClaw — Claw agent skill injection and cloud session data collection."""


@skillclaw.command()
def setup():
    """Interactive first-time configuration wizard."""
    from .setup_wizard import SetupWizard

    SetupWizard().run()


@skillclaw.command()
@click.option(
    "--port",
    type=int,
    default=None,
    help="Override proxy port for this session.",
)
@click.option(
    "--daemon",
    "-d",
    is_flag=True,
    default=False,
    help="Run SkillClaw in the background.",
)
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Log file used with --daemon (default: ~/.skillclaw/skillclaw.log).",
)
def start(port: int | None, daemon: bool, log_file: str | None):
    """Start SkillClaw (proxy + skill injection + optional PRM)."""
    import asyncio

    from .log_color import setup_logging

    setup_logging()

    cs = ConfigStore()
    if not cs.exists():
        click.echo(
            "No config found. Run 'skillclaw setup' first.",
            err=True,
        )
        sys.exit(1)

    if daemon:
        pid, log_path = _spawn_daemon_process(
            port,
            log_file,
            effective_port=_effective_proxy_port(cs, port),
        )
        click.echo(
            f"SkillClaw started in background (PID={pid}). Logs: {log_path}. "
            "Use 'skillclaw status' to check health and 'skillclaw stop' to stop it."
        )
        return

    if port:
        import tempfile

        import yaml

        from .config_store import ConfigStore as _CS

        data = cs.load()
        data.setdefault("proxy", {})["port"] = port
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            yaml.dump(data, tmp)
        finally:
            tmp.close()
        tmp_path = Path(tmp.name)
        cs = _CS(config_file=tmp_path)
    else:
        tmp_path = None

    from .launcher import SkillClawLauncher

    launcher = SkillClawLauncher(cs)
    try:
        asyncio.run(launcher.start())
    except KeyboardInterrupt:
        click.echo("\nInterrupted — stopping SkillClaw.")
        launcher.stop()
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@skillclaw.command()
def stop():
    """Stop a running SkillClaw instance."""
    import os
    import signal
    from pathlib import Path

    pid_file = Path.home() / ".skillclaw" / "skillclaw.pid"
    if not pid_file.exists():
        click.echo("SkillClaw is not running (no PID file found).")
        return
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        pid_file.unlink(missing_ok=True)
        click.echo(f"Sent SIGTERM to PID {pid}.")
    except ProcessLookupError:
        click.echo("Process not found — cleaning up stale PID file.")
        pid_file.unlink(missing_ok=True)
    except Exception as e:
        click.echo(f"Error stopping SkillClaw: {e}", err=True)


@skillclaw.command()
def status():
    """Check whether SkillClaw is running."""
    pid = _read_pid()
    if pid is None:
        click.echo("SkillClaw: not running")
        return

    if not _is_process_alive(pid):
        click.echo("SkillClaw: not running (stale PID file)")
        _clear_pid()
        return

    cs = ConfigStore()
    port = int(cs.get("proxy.port") or 30000)

    healthy = _healthz_ready(port, timeout=2.0)
    if healthy:
        click.echo(f"SkillClaw: running  (PID={pid}, proxy=:{port})")
    else:
        click.echo(f"SkillClaw: starting (PID={pid}, proxy=:{port})")


@skillclaw.command(name="config")
@click.argument("key_or_action")
@click.argument("value", required=False)
def config_cmd(key_or_action: str, value: str | None):
    """Get or set a config value.

    Examples:\n
      skillclaw config show\n
      skillclaw config proxy.port 30001
    """
    cs = ConfigStore()
    if key_or_action == "show":
        if not cs.exists():
            click.echo("No config file found. Run 'skillclaw setup' first.")
            return
        click.echo(f"Config file: {CONFIG_FILE}\n")
        click.echo(cs.describe())
        return

    if value is None:
        result = cs.get(key_or_action)
        if result is None:
            click.echo(f"{key_or_action}: (not set)")
        else:
            click.echo(f"{key_or_action}: {result}")
        return

    cs.set(key_or_action, value)
    click.echo(f"Set {key_or_action} = {cs.get(key_or_action)}")


@skillclaw.group()
def doctor():
    """Integration diagnostics."""


@doctor.command(name="hermes")
def doctor_hermes():
    """Inspect the local Hermes integration state."""
    from .claw_adapter import inspect_hermes_config

    cs = ConfigStore()
    if not cs.exists():
        raise click.ClickException("No config file found. Run 'skillclaw setup' first.")

    report = inspect_hermes_config(cs.to_skillclaw_config())
    _echo_report(report)


@doctor.command(name="codex")
def doctor_codex():
    """Inspect the local Codex integration state."""
    from .claw_adapter import inspect_codex_config

    cs = ConfigStore()
    if not cs.exists():
        raise click.ClickException("No config file found. Run 'skillclaw setup' first.")

    report = inspect_codex_config(cs.to_skillclaw_config())
    _echo_report(report)


@doctor.command(name="claude")
def doctor_claude():
    """Inspect the local Claude Code integration state."""
    from .claw_adapter import inspect_claude_config

    cs = ConfigStore()
    if not cs.exists():
        raise click.ClickException("No config file found. Run 'skillclaw setup' first.")

    report = inspect_claude_config(cs.to_skillclaw_config())
    _echo_report(report)


@skillclaw.group()
def restore():
    """Restore agent integration state from backups."""


@restore.command(name="hermes")
@click.option(
    "--backup",
    "backup_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Restore from a specific backup file instead of the latest Hermes backup.",
)
def restore_hermes(backup_path: str | None):
    """Restore ~/.hermes/config.yaml from a saved backup."""
    from .claw_adapter import restore_hermes_config

    try:
        result = restore_hermes_config(Path(backup_path).expanduser() if backup_path else None)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from None

    click.echo(f"Restored Hermes config: {result['target']} <- {result['source']}")


@restore.command(name="codex")
@click.option(
    "--backup",
    "backup_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Restore from a specific backup file instead of the latest Codex backup.",
)
def restore_codex(backup_path: str | None):
    """Restore ~/.codex/config.toml from a saved backup."""
    from .claw_adapter import restore_codex_config

    try:
        result = restore_codex_config(Path(backup_path).expanduser() if backup_path else None)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from None

    click.echo(f"Restored Codex config: {result['target']} <- {result['source']}")


@restore.command(name="claude")
@click.option(
    "--backup",
    "backup_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Restore from a specific backup file instead of the latest Claude Code backup.",
)
def restore_claude(backup_path: str | None):
    """Restore ~/.claude/settings.json from a saved backup."""
    from .claw_adapter import restore_claude_config

    try:
        result = restore_claude_config(Path(backup_path).expanduser() if backup_path else None)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from None

    click.echo(f"Restored Claude Code settings: {result['target']} <- {result['source']}")


@skillclaw.group()
def validation():
    """Background validation commands."""


@validation.command(name="status")
def validation_status():
    """Show background validation configuration and current availability."""
    from .validation_worker import ValidationWorker

    cs = ConfigStore()
    cfg = cs.to_skillclaw_config()
    worker = ValidationWorker(cfg)
    snapshot = worker.status_snapshot()
    for key, value in snapshot.items():
        click.echo(f"{key}: {value}")


@validation.command(name="run-once")
@click.option("--force", is_flag=True, help="Run one validation poll even if the client is not idle.")
def validation_run_once(force: bool):
    """Run one background validation polling iteration."""
    import asyncio

    from .validation_worker import ValidationWorker

    cs = ConfigStore()
    cfg = cs.to_skillclaw_config()
    worker = ValidationWorker(cfg)
    result = asyncio.run(worker.run_once(force=force))
    for key, value in result.items():
        click.echo(f"{key}: {value}")


@skillclaw.group()
def skills():
    """Skill management commands."""


def _sharing_backend(cfg) -> str:
    backend = str(getattr(cfg, "sharing_backend", "") or "").strip().lower()
    if backend:
        return backend
    if getattr(cfg, "sharing_local_root", ""):
        return "local"
    return "oss"


def _sharing_target(cfg) -> str:
    backend = _sharing_backend(cfg)
    group = getattr(cfg, "sharing_group_id", "default")
    if backend == "local":
        return f"local storage ({cfg.sharing_local_root}/{group})"
    bucket = getattr(cfg, "sharing_bucket", "")
    endpoint = getattr(cfg, "sharing_endpoint", "")
    target = f"{bucket}/{group}" if bucket else group
    if endpoint:
        return f"{backend} storage ({target} @ {endpoint})"
    return f"{backend} storage ({target})"


def _require_sharing(cs: ConfigStore):
    """Validate that sharing is enabled and configured. Returns (cfg, SkillHub) or raises."""
    cfg = cs.to_skillclaw_config()
    if not cfg.sharing_enabled:
        raise click.ClickException(
            "Skill sharing is not enabled. "
            "Run 'skillclaw config sharing.enabled true' or 'skillclaw setup' to configure."
        )
    backend = _sharing_backend(cfg)
    if backend == "local":
        if not cfg.sharing_local_root:
            raise click.ClickException("Local sharing backend is not configured. Set sharing.local_root first.")
    elif backend == "s3":
        if not cfg.sharing_bucket:
            raise click.ClickException("S3 bucket is not configured. Set sharing.bucket first.")
        if not cfg.sharing_access_key_id or not cfg.sharing_secret_access_key:
            raise click.ClickException(
                "S3 credentials are not configured. Set sharing.access_key_id and sharing.secret_access_key."
            )
    elif backend == "oss":
        if not cfg.sharing_endpoint or not cfg.sharing_bucket:
            raise click.ClickException(
                "OSS endpoint or bucket is not configured. Set sharing.endpoint and sharing.bucket first."
            )
        if not cfg.sharing_access_key_id or not cfg.sharing_secret_access_key:
            raise click.ClickException(
                "OSS credentials are not configured. Set sharing.access_key_id and sharing.secret_access_key."
            )
    else:
        raise click.ClickException("Sharing backend is not configured. Set sharing.backend to local, s3, or oss.")
    from .skill_hub import SkillHub

    hub = SkillHub.from_config(cfg)
    return cfg, hub


@skills.command(name="push")
@click.option("--no-filter", is_flag=True, help="Skip effectiveness quality gate.")
def skills_push(no_filter):
    """Push local skills to the shared cloud."""
    cs = ConfigStore()
    cfg, hub = _require_sharing(cs)
    click.echo(f"Pushing skills to {_sharing_target(cfg)} ...")
    skill_filter = None
    if not no_filter:
        stats_path = os.path.join(cfg.skills_dir, "skill_stats.json")
        if os.path.exists(stats_path):
            import json

            try:
                with open(stats_path, encoding="utf-8") as f:
                    stats = json.load(f)
                skill_filter = {
                    "stats": stats,
                    "min_injections": cfg.sharing_push_min_injections,
                    "min_effectiveness": cfg.sharing_push_min_effectiveness,
                }
            except Exception:
                pass
    result = hub.push_skills(cfg.skills_dir, skill_filter=skill_filter)
    click.echo(
        f"Done: {result['uploaded']} uploaded, "
        f"{result['skipped']} unchanged, "
        f"{result.get('filtered', 0)} filtered, "
        f"{result['total_local']} total local skills."
    )


@skills.command(name="pull")
def skills_pull():
    """Pull shared skills from the cloud."""
    cs = ConfigStore()
    cfg, hub = _require_sharing(cs)
    click.echo(f"Pulling skills from {_sharing_target(cfg)} ...")
    result = hub.pull_skills(cfg.skills_dir)
    msg = (
        f"Done: {result['downloaded']} downloaded, "
        f"{result['skipped']} unchanged, "
        f"{result.get('deleted', 0)} deleted, "
        f"{result['total_remote']} total remote skills."
    )
    if result.get("restored_from_backup"):
        msg += f" Restored from backup: {result.get('backup_dir', '')}"
    click.echo(msg)


@skills.command(name="sync")
def skills_sync():
    """Bidirectional sync: pull then push."""
    cs = ConfigStore()
    cfg, hub = _require_sharing(cs)
    click.echo(f"Syncing skills with {_sharing_target(cfg)} ...")
    result = hub.sync_skills(cfg.skills_dir)
    pr = result["pull"]
    ps = result["push"]
    click.echo(
        f"Pull: {pr['downloaded']} downloaded, {pr['skipped']} unchanged\n"
        f"Push: {ps['uploaded']} uploaded, {ps['skipped']} unchanged"
    )


@skills.command(name="list-remote")
def skills_list_remote():
    """List skills available in the shared storage backend."""
    cs = ConfigStore()
    cfg, hub = _require_sharing(cs)
    remote = hub.list_remote()
    if not remote:
        click.echo("No skills found on the cloud.")
        return
    click.echo(f"\n{'=' * 60}")
    click.echo(f"  Shared Skills ({len(remote)} total)")
    click.echo(f"{'=' * 60}\n")
    for rec in sorted(remote, key=lambda r: r.get("name", "")):
        name = rec.get("name", "?")
        desc = rec.get("description", "")
        cat = rec.get("category", "general")
        by = rec.get("uploaded_by", "?")
        at = rec.get("uploaded_at", "?")
        click.echo(f"  {name}  [{cat}]")
        if desc:
            click.echo(f"    {desc}")
        click.echo(f"    by {by}  at {at}")
        click.echo()


if __name__ == "__main__":
    skillclaw()
