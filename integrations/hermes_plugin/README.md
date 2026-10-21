# Hermes plugin: session capture without the proxy

The default Hermes integration (`skillclaw setup` → `hermes`) rewrites `~/.hermes/config.yaml`
so that every model request goes through the SkillClaw proxy. That works when Hermes uses an
API key for one OpenAI-compatible / OpenRouter / Bedrock upstream. It does not work when:

- Hermes uses a **subscription login** (ChatGPT/Codex OAuth, Claude subscription) — the proxy
  forwards with a static `llm_api_key` and cannot refresh those tokens;
- several **Hermes profiles** use different providers (the proxy has one upstream);
- you rely on Hermes' own **provider fallback chain**.

This plugin is an alternative capture path. Hermes keeps its own provider routing; the plugin
only observes finished turns via the `post_llm_call` hook and writes sessions into local shared
storage in the format `skillclaw-evolve-server` already reads:

```
<local_root>/<group_id>/sessions/<session>.json   # num_turns + turns[]
```

Each turn carries `read_skills` / `modified_skills` taken from Hermes' `skill_view` /
`skill_manage` calls, so the evolve server can group sessions by skill. These are **attempted**
calls, not proof of successful reads or committed writes (denied/staged/failed calls may appear).
`skill_view` accepts `name` and optional `file_path`; parallel reads are separate `tool_calls`.
`skill_manage` uses `operations[]`; the legacy flat `action`/`name` form is also accepted.
No tool result or skill body is copied into the record.

Turns accumulate until the session is idle (default 30 min), Hermes finalizes/resets it,
or an episode reaches 40 turns / 1 MiB. Later turns start another uniquely keyed episode.
A daemon worker checks for idle records at most every 30 seconds, **without requiring another
turn**. Unloading the plugin stops the worker; finalization/reset publishes only the named
session. `on_session_end` is deliberately not used: it fires after individual runs/turns,
not just when a conversation closes.

## Install

```bash
cp -r integrations/hermes_plugin/skillclaw_capture ~/.hermes/plugins/skillclaw-capture
# per profile: ~/.hermes/profiles/<name>/plugins/skillclaw-capture
```

Export these variables **only in the intended profile's process environment** (or its
profile-local `.env` if your launcher loads it). Setting `SKILLCLAW_CAPTURE_ROOT` is explicit
opt-in to capture user/assistant text; without it, registration has no hooks, worker, or writes.
Avoid exporting it globally to unrelated profiles.

```bash
export SKILLCLAW_CAPTURE_ROOT=~/.skillclaw/local-share   # same as the server's local root
export SKILLCLAW_GROUP_ID=my-group
# optional
export SKILLCLAW_USER_ALIAS=alice
export SKILLCLAW_CAPTURE_IDLE_SECONDS=1800
```

Then enable it: `hermes plugins enable skillclaw-capture`.

Run the evolve server against the same root, with its own LLM endpoint:

```bash
skillclaw-evolve-server --storage-backend local --local-root ~/.skillclaw/local-share --group-id my-group
```

Do **not** also run the proxy integration for the same Hermes profile, or turns are recorded twice.

## What is stored

User message and final assistant text of each turn (best-effort secret redaction), model/platform,
tool **names**, skill names and optional skill-reference paths. The plugin does not copy system
prompts, earlier history, arbitrary tool arguments or tool results. Text supplied as user/final
assistant messages can itself contain sensitive material: **redaction is not anonymization and
cannot guarantee removal of credentials or personal/confidential data**. The evolve server may
send these records to its own configured LLM; do not enable capture for data that endpoint must
not receive. This plugin itself makes no network requests.

Profile identity uses the resolved Hermes home, not just the display alias. Open records and
receipts are isolated under `<root>/.hermes-capture/<group>/<profile-and-alias-hash>/`; identical
session/turn IDs from separate homes do not collide or flush one another. **The queue is still
shared by group**: use different groups or roots for private/work data, with appropriate filesystem
permissions. Group/alias must start with a letter, digit, `_` or `-`, then contain only letters,
digits, `_`, `-`, `.` (maximum 80 characters). Idle seconds must be finite and positive.

## Durability and compatibility

- Requires POSIX local storage with `flock`, directory `fsync`, and atomic rename. State and queue
  must be on the same filesystem. New directories/files use 0700/0600; existing root permissions
  are not changed. Use a trusted root, not one writable by untrusted users or containing symlinks
  into other storage. Windows and network filesystem semantics are not supported here.
- A record is atomically committed before its dedupe receipt; recovery also checks the open
  record's turn hashes. Before publishing, all receipts are made durable, then the open file is
  **moved**, not copied, into the queue. This avoids the process-crash copy/unlink window even if a
  consumer immediately removes the queue file. Receipts are retained indefinitely; deleting them
  permits old hook deliveries to be captured again. This is not an end-to-end exactly-once promise
  for downstream consumers or a filesystem-independent power-loss guarantee.
- Invalid JSON/schema is preserved in place and warned about, never overwritten or deleted.
  Other sessions can still flush. Inspect/recover the damaged file manually before retrying that
  session. Failed writes/handoffs retain committed open state; a turn failing before its first
  durable write requires hook redelivery (Hermes does not guarantee this).
- After abrupt process exit, open records remain until this opted-in plugin is loaded again.
  No worker can flush while Hermes is not running. Plugin unload stops capture and leaves pending
  records for a later load; changing root/group/alias/home does not migrate existing state.
- Pre-hardening state directly under `.hermes-capture/<group>/open` is intentionally not imported:
  its owning profile cannot be inferred safely. Finalize it with the old installation or recover
  it manually before upgrade. Existing queue records are left untouched.
- Contracts verified against Hermes `e726b79803b0b63d126f9836da0ba3a7f09827f1`:
  `agent/turn_finalizer.py` (`post_llm_call`), `hermes_cli/lifecycle.py`,
  `hermes_cli/plugins.py` (`ctx.on_unload`), `tools/skills_tool.py` and
  `tools/skill_manager_tool.py`. See the upstream
  [observer-hook guide](https://hermes-agent.nousresearch.com/docs/developer-guide/observer-hooks/).
  Older Hermes versions lacking these hooks or `ctx.on_unload` require an upgrade.
