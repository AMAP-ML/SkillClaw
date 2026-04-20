# Adapted from MetaClaw
"""
Skill Manager for SkillClaw skill retrieval, injection, and local persistence.

Loads skills from a directory of skill subdirectories using the AgentSkills-
compatible format (shared with OpenClaw / Pi coding-agent):

    memory_data/skills/
        skill-name/
            SKILL.md           <- YAML frontmatter + markdown body
            scripts/           <- optional executable scripts
            references/        <- optional reference docs
            assets/            <- optional assets (templates, images, etc.)
        another-skill/
            SKILL.md

Each SKILL.md must have YAML frontmatter with at least ``name`` and
``description``::

    ---
    name: debug-systematically
    description: "Use when diagnosing a bug. Gather evidence before forming
      hypotheses.  NOT for: simple typo fixes."
    metadata:
      {
        "openclaw": { "emoji": "🔍" },
        "skillclaw": { "category": "coding" }
      }
    ---

    # Debug Systematically
    ...

Frontmatter fields
------------------
Required:
  name         lowercase-hyphenated slug
  description  rich trigger description — what, when to use, when NOT to use

Optional:
  metadata     nested dict; ``openclaw`` block for OpenClaw gating/install,
               ``skillclaw`` block for SkillClaw-specific fields (``category``)
  category     (legacy) — if present and ``metadata.skillclaw.category`` is
               absent, used as fallback; defaults to ``"general"``

Valid categories: general, coding, research, data_analysis, security,
                  communication, automation, agentic, productivity, common_mistakes
"""

import glob
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,63}$")

# ------------------------------------------------------------------ #
# Frontmatter parser                                                   #
# ------------------------------------------------------------------ #

_CORE_FM_KEYS = {"name", "description", "metadata", "category"}


def _parse_skill_md(path: str) -> Optional[Dict[str, Any]]:
    """Parse a SKILL.md file (AgentSkills / OpenClaw compatible format).

    Returns a dict with keys: name, description, category, content, and
    optionally metadata.  Extra frontmatter fields (e.g. ``homepage``,
    ``user-invocable``) are preserved verbatim so they survive round-trip
    writes.  Returns ``None`` if the file is missing required fields or
    has no frontmatter.

    Category resolution order:
      1. ``metadata.skillclaw.category``
      2. top-level ``category`` (legacy)
      3. ``"general"`` (default)
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    except OSError as e:
        logger.warning("[SkillManager] could not read %s: %s", path, e)
        return None

    if not raw.startswith("---"):
        return None

    end_idx = raw.find("\n---", 3)
    if end_idx == -1:
        return None

    fm_text = raw[3:end_idx].strip()
    body = raw[end_idx + 4 :].strip()

    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        logger.warning("[SkillManager] invalid YAML frontmatter in %s", path)
        fm = {}

    if not isinstance(fm, dict):
        return None

    name = str(fm.get("name", "")).strip()
    description = str(fm.get("description", "")).strip()

    if not name or not description:
        logger.warning("[SkillManager] skipping %s — missing name or description", path)
        return None

    metadata = fm.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        metadata = None

    # Resolve category: metadata.skillclaw.category > top-level category > "general"
    category = "general"
    sc_meta = (metadata or {}).get("skillclaw", {})
    if isinstance(sc_meta, dict) and sc_meta.get("category"):
        category = str(sc_meta["category"]).strip()
    elif fm.get("category"):
        category = str(fm["category"]).strip()

    result: Dict[str, Any] = {
        "id": hashlib.sha256(name.encode()).hexdigest()[:12],
        "name": name,
        "description": description,
        "category": category,
        "content": body,
        "file_path": os.path.realpath(path),
    }
    if metadata:
        result["metadata"] = metadata

    # Preserve extra OpenClaw frontmatter fields (homepage, user-invocable, etc.)
    extra = {k: v for k, v in fm.items() if k not in _CORE_FM_KEYS}
    if extra:
        result["_extra_frontmatter"] = extra

    return result


# ------------------------------------------------------------------ #
# SkillManager                                                         #
# ------------------------------------------------------------------ #


class SkillManager:
    """Loads skills from a directory of AgentSkills / OpenClaw compatible
    skill folders.

    Each skill is a subdirectory containing a ``SKILL.md`` with YAML
    frontmatter (``name``, ``description``, optional ``metadata``) and a
    Markdown body.

    Supports two retrieval modes:
      * ``"template"`` – flat effectiveness-ranked retrieval, zero latency
      * ``"embedding"`` – cosine similarity via SentenceTransformer
    """

    def __init__(
        self,
        skills_dir: str,
        public_skill_root: str = "",
        retrieval_mode: str = "template",
        embedding_model_path: Optional[str] = None,
    ):
        if retrieval_mode not in ("template", "embedding"):
            raise ValueError(f"retrieval_mode must be 'template' or 'embedding', got '{retrieval_mode}'")
        if not os.path.isdir(skills_dir):
            raise FileNotFoundError(f"Skills directory not found: {skills_dir}")

        self._skills_dir = skills_dir
        self._public_skill_root = public_skill_root.strip()
        self.retrieval_mode = retrieval_mode
        self.embedding_model_path = embedding_model_path or "Qwen/Qwen3-Embedding-0.6B"

        self._embedding_model = None
        self._skill_embeddings_cache: Optional[Dict] = None

        # Monotonically-increasing counter. Incremented whenever the local
        # skill library changes so callers can drop stale snapshots.
        self.generation: int = 0

        self.skills = self._load_skills()
        self._skills_fingerprint = self._compute_skills_fingerprint()
        self._stats = self._load_stats()
        self._stats_dirty = 0

        counts = self._category_counts()
        logger.info(
            "[SkillManager] loaded %d skills from %s | mode=%s | categories=%s",
            len(self.skills.get("all_skills", [])),
            skills_dir,
            retrieval_mode,
            dict(counts),
        )

        if retrieval_mode == "embedding":
            self._compute_skill_embeddings()

    # ------------------------------------------------------------------ #
    # Skill stats tracking                                                 #
    # ------------------------------------------------------------------ #

    def _stats_path(self) -> str:
        return os.path.join(self._skills_dir, "skill_stats.json")

    def _load_stats(self) -> Dict[str, Dict[str, Any]]:
        path = self._stats_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[SkillManager] failed to load stats: %s", e)
            return {}

    def _save_stats(self) -> None:
        try:
            with open(self._stats_path(), "w", encoding="utf-8") as f:
                json.dump(self._stats, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning("[SkillManager] failed to save stats: %s", e)

    def _maybe_flush_stats(self) -> None:
        """Persist stats every 10 mutations to avoid excessive I/O."""
        self._stats_dirty += 1
        if self._stats_dirty >= 10:
            self._save_stats()
            self._stats_dirty = 0

    def record_injection(self, skill_names: list[str]) -> None:
        """Record that these skills were injected into a request."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        for name in skill_names:
            entry = self._stats.setdefault(
                name,
                {
                    "inject_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "last_injected_at": "",
                    "effectiveness": 0.5,
                },
            )
            entry["inject_count"] += 1
            entry["last_injected_at"] = now
        self._maybe_flush_stats()

    def record_feedback(self, skill_names: list[str], score: float) -> None:
        """Record PRM feedback for skills that were injected in a turn."""
        for name in skill_names:
            entry = self._stats.get(name)
            if entry is None:
                continue
            if score > 0:
                entry["positive_count"] += 1
            elif score < 0:
                entry["negative_count"] += 1
            else:
                entry["neutral_count"] += 1
            total = entry["inject_count"]
            entry["effectiveness"] = entry["positive_count"] / total if total > 0 else 0.5
        self._maybe_flush_stats()

    def get_effectiveness(self, skill_name: str) -> float:
        """Return the effectiveness score for a skill (default 0.5 for unknown)."""
        entry = self._stats.get(skill_name)
        if entry is None:
            return 0.5
        return entry.get("effectiveness", 0.5)

    def get_stats_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of the full stats dict."""
        return dict(self._stats)

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    def _load_skills(self) -> Dict[str, Any]:
        """Scan skills_dir for */SKILL.md files and parse each into a flat collection."""
        result: Dict[str, Any] = {"all_skills": []}

        paths = self._skill_md_paths()
        if not paths:
            logger.warning("[SkillManager] no SKILL.md files found in %s", self._skills_dir)
            return result

        for path in paths:
            skill = _parse_skill_md(path)
            if skill is None:
                continue
            result["all_skills"].append(skill)

        return result

    def _skill_md_paths(self) -> list[str]:
        if self._is_hermes_skill_root():
            pattern = os.path.join(self._skills_dir, "**", "SKILL.md")
            return sorted(glob.glob(pattern, recursive=True))
        pattern = os.path.join(self._skills_dir, "*", "SKILL.md")
        return sorted(glob.glob(pattern))

    def _compute_skills_fingerprint(self) -> tuple[tuple[str, int, int], ...]:
        fingerprint: list[tuple[str, int, int]] = []
        for path in self._skill_md_paths():
            try:
                stat = os.stat(path)
            except OSError:
                continue
            fingerprint.append(
                (
                    os.path.realpath(path),
                    int(stat.st_mtime_ns),
                    int(stat.st_size),
                )
            )
        return tuple(fingerprint)

    def _is_hermes_skill_root(self) -> bool:
        return os.path.realpath(self._skills_dir) == os.path.realpath(
            os.path.join(os.path.expanduser("~"), ".hermes", "skills")
        )

    def _skill_dir_path(self, skill: dict) -> str:
        name = str(skill.get("name", "unknown") or "unknown").strip()
        category = str(skill.get("category", "general") or "general").strip()
        if self._is_hermes_skill_root() and category and category != "general":
            return os.path.join(self._skills_dir, category, name)
        return os.path.join(self._skills_dir, name)

    def _skill_md_path(self, skill: dict) -> str:
        return os.path.join(self._skill_dir_path(skill), "SKILL.md")

    def reload(self) -> None:
        """Re-scan the skills directory and rebuild the internal skill bank."""
        self._save_stats()
        self.skills = self._load_skills()
        self._skills_fingerprint = self._compute_skills_fingerprint()
        self._stats = self._load_stats()
        self._skill_embeddings_cache = None
        if self.retrieval_mode == "embedding":
            self._compute_skill_embeddings()
        logger.info("[SkillManager] reloaded skills from %s", self._skills_dir)

    def refresh_if_changed(self) -> bool:
        """Reload skills if the on-disk skill library changed externally."""
        current = self._compute_skills_fingerprint()
        if current == self._skills_fingerprint:
            return False
        self.reload()
        self.generation += 1
        logger.info("[SkillManager] detected local skill changes; refreshed library")
        return True

    # ------------------------------------------------------------------ #
    # Embedding helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_embedding_model(self):
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding retrieval. "
                    "Install with: pip install sentence-transformers"
                )
            logger.info("[SkillManager] loading embedding model: %s", self.embedding_model_path)
            self._embedding_model = SentenceTransformer(self.embedding_model_path)
        return self._embedding_model

    @staticmethod
    def _skill_to_text(skill: Dict[str, Any]) -> str:
        parts = [
            skill.get("name", "").strip(),
            skill.get("description", "").strip(),
        ]
        content = skill.get("content", "").strip()
        if content:
            parts.append(content[:200])
        return ". ".join(p for p in parts if p)

    def _compute_skill_embeddings(self) -> Dict:
        if self._skill_embeddings_cache is not None:
            return self._skill_embeddings_cache

        import numpy as np

        all_items = list(self.skills.get("all_skills", []))
        texts = [self._skill_to_text(skill) for skill in all_items]

        if not texts:
            self._skill_embeddings_cache = {
                "items": [],
                "embeddings": np.zeros((0, 0), dtype=float),
            }
            return self._skill_embeddings_cache

        model = self._get_embedding_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self._skill_embeddings_cache = {
            "items": all_items,
            "embeddings": embeddings,
        }
        logger.info("[SkillManager] cached %d skill embeddings", len(all_items))
        return self._skill_embeddings_cache

    def _weighted_score(self, similarity: float, skill_name: str) -> float:
        """Combine embedding similarity with effectiveness for ranking."""
        eff = self.get_effectiveness(skill_name)
        return similarity * (0.3 + 0.7 * eff)

    def _deduplicate_by_embedding(
        self, indices: list[int], cache: dict, top_k: int, threshold: float = 0.9
    ) -> list[int]:
        """Remove near-duplicate skills from candidate list, keeping the one
        with higher weighted score. Fills freed slots with next candidates."""
        if not indices:
            return []
        embs = cache["embeddings"]
        kept: list[int] = []
        for idx in indices:
            is_dup = False
            for kept_idx in kept:
                sim = float(embs[idx] @ embs[kept_idx])
                if sim > threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(idx)
            if len(kept) >= top_k:
                break
        return kept

    def _embedding_retrieve(self, task_description: str, top_k: int) -> list[dict]:
        cache = self._compute_skill_embeddings()
        if not cache["items"]:
            return []
        model = self._get_embedding_model()
        query_emb = model.encode(
            [task_description],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]

        sims = cache["embeddings"] @ query_emb
        items = cache["items"]

        ranked = sorted(
            range(len(sims)),
            key=lambda i: self._weighted_score(float(sims[i]), items[i].get("name", "")),
            reverse=True,
        )
        candidate_indices = ranked[: max(top_k * 3, top_k)]
        kept = self._deduplicate_by_embedding(candidate_indices, cache, top_k)
        return [items[i] for i in kept]

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def retrieve(self, task_description: str, top_k: int = 6) -> list[dict]:
        """Retrieve relevant skills for *task_description*.

        In embedding mode, skills are ranked by similarity * effectiveness and
        deduplicated by embedding similarity. In template mode, all skills are
        sorted by effectiveness without any category-specific routing.
        """
        if self.retrieval_mode == "embedding":
            return self._embedding_retrieve(task_description, top_k)

        all_skills = list(self.skills.get("all_skills", []))
        all_skills_sorted = sorted(
            all_skills,
            key=lambda s: self.get_effectiveness(s.get("name", "")),
            reverse=True,
        )
        return all_skills_sorted[:top_k]

    # ------------------------------------------------------------------ #
    # Skill catalog (OpenClaw-compatible injection)                        #
    # ------------------------------------------------------------------ #

    def get_all_skills(self) -> list[dict]:
        """Return a flat list of ALL loaded skills eligible for model invocation.

        Skills with ``disable-model-invocation: true`` in their frontmatter
        are filtered out (matching OpenClaw behaviour).
        """
        return [
            s
            for s in self.skills.get("all_skills", [])
            if not s.get("_extra_frontmatter", {}).get("disable-model-invocation", False)
        ]

    def get_skill_path_map(self) -> Dict[str, Dict[str, str]]:
        """Return a mapping from file_path → {skill_id, skill_name}.

        Used by the server to resolve which skill a ``read`` tool call targets.
        """
        path_map: Dict[str, Dict[str, str]] = {}
        for s in self.get_all_skills():
            for fp in [s.get("file_path", ""), self._public_skill_path(s)]:
                if fp:
                    path_map[fp] = {
                        "skill_id": s.get("id", ""),
                        "skill_name": s.get("name", ""),
                    }
        return path_map

    def _public_skill_path(self, skill: dict) -> str:
        if not self._public_skill_root:
            return ""
        name = str(skill.get("name", "")).strip()
        if not name:
            return ""
        return os.path.join(self._public_skill_root, name, "SKILL.md")

    @staticmethod
    def _escape_xml(text: str) -> str:
        """Escape XML special characters (matching OpenClaw's escapeXml)."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    def format_skills_for_prompt(self, skills: list[dict]) -> str:
        """Build the *full* ``<available_skills>`` XML catalog.

        Output matches ``formatSkillsForPrompt`` from
        ``@mariozechner/pi-coding-agent`` (used by OpenClaw): preamble text
        followed by ``<available_skills>`` with ``<name>``, ``<description>``,
        and ``<location>`` per skill.
        """
        if not skills:
            return ""
        escape = SkillManager._escape_xml
        lines = [
            "\n\nThe following skills provide specialized instructions for specific tasks.",
            "Use the read tool to load a skill's file when the task matches its description.",
            "When a skill file references a relative path, resolve it against the skill "
            "directory (parent of SKILL.md / dirname of the path) and use that absolute "
            "path in tool commands.",
            "",
            "<available_skills>",
        ]
        for skill in skills:
            lines.append("  <skill>")
            lines.append(f"    <name>{escape(skill.get('name', ''))}</name>")
            lines.append(f"    <description>{escape(skill.get('description', ''))}</description>")
            lines.append(
                f"    <location>{escape(self._public_skill_path(skill) or skill.get('file_path', ''))}</location>"
            )
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def format_skills_compact(self, skills: list[dict]) -> str:
        """Build the *compact* ``<available_skills>`` XML catalog.

        Omits ``<description>`` to save tokens.  Matches OpenClaw's
        ``formatSkillsCompact``.
        """
        if not skills:
            return ""
        escape = SkillManager._escape_xml
        lines = [
            "\n\nThe following skills provide specialized instructions for specific tasks.",
            "Use the read tool to load a skill's file when the task matches its name.",
            "When a skill file references a relative path, resolve it against the skill "
            "directory (parent of SKILL.md / dirname of the path) and use that absolute "
            "path in tool commands.",
            "",
            "<available_skills>",
        ]
        for skill in skills:
            lines.append("  <skill>")
            lines.append(f"    <name>{escape(skill.get('name', ''))}</name>")
            lines.append(
                f"    <location>{escape(self._public_skill_path(skill) or skill.get('file_path', ''))}</location>"
            )
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    @staticmethod
    def build_skills_section(
        skills_prompt: str,
        read_tool_name: str = "read",
    ) -> str:
        """Wrap a skills catalog string with the ``## Skills (mandatory)``
        instruction block.

        Matches OpenClaw's ``buildSkillsSection`` in ``system-prompt.ts``.
        """
        trimmed = skills_prompt.strip()
        if not trimmed:
            return ""
        return "\n".join(
            [
                "## Skills (mandatory)",
                "Before replying: scan <available_skills> <description> entries.",
                f"- If exactly one skill clearly applies: read its SKILL.md at "
                f"<location> with `{read_tool_name}`, then follow it.",
                "- If multiple could apply: choose the most specific one, then read/follow it.",
                "- If none clearly apply: do not read any SKILL.md.",
                "Constraints: never read more than one skill up front; only read after selecting.",
                "- When a skill drives external API writes, assume rate limits: prefer fewer "
                "larger writes, avoid tight one-item loops, serialize bursts when possible, "
                "and respect 429/Retry-After.",
                trimmed,
                "",
            ]
        )

    def build_injection_prompt(
        self,
        max_chars: int = 30_000,
        read_tool_name: str = "read",
    ) -> str:
        """One-call helper: catalog all skills and wrap with instructions.

        Uses the full format (name + description + location) when the catalog
        fits within *max_chars*; falls back to compact format (name + location)
        otherwise.  Returns the empty string when no skills are loaded.
        """
        skills = self.get_all_skills()
        if not skills:
            return ""
        full_prompt = self.format_skills_for_prompt(skills)
        if len(full_prompt) <= max_chars:
            catalog = full_prompt
        else:
            catalog = self.format_skills_compact(skills)
        return self.build_skills_section(catalog, read_tool_name)

    def _remove_skill_from_memory(self, name: str) -> None:
        """Remove a skill from in-memory structures (not from disk)."""
        self.skills["all_skills"] = [s for s in self.skills.get("all_skills", []) if s.get("name") != name]

    def add_skill(self, skill: dict) -> bool:
        """
        Add a new skill to the in-memory bank and write its SKILL.md file.

        Returns True if the skill was added, False if it already exists
        (unless ``skill["_replace"]`` is truthy, in which case the old
        version is overwritten).
        """
        name = skill.get("name", "").strip()
        if not name:
            logger.warning("[SkillManager] add_skill called with missing name")
            return False
        if not _SAFE_NAME_RE.match(name):
            logger.warning("[SkillManager] rejected invalid skill name: %s", name)
            return False

        existing = self._get_all_skill_names()
        if name in existing:
            if skill.get("_replace"):
                self._remove_skill_from_memory(name)
                logger.info("[SkillManager] replacing existing skill: %s", name)
            else:
                logger.info("[SkillManager] skipping duplicate skill: %s", name)
                return False

        clean_skill = {k: v for k, v in skill.items() if not k.startswith("_") or k == "_extra_frontmatter"}
        if "id" not in clean_skill:
            clean_skill["id"] = hashlib.sha256(name.encode()).hexdigest()[:12]
        if "file_path" not in clean_skill:
            clean_skill["file_path"] = os.path.realpath(self._skill_md_path(clean_skill))
        clean_skill["category"] = str(clean_skill.get("category", "general") or "general").strip()
        self.skills.setdefault("all_skills", []).append(clean_skill)

        self._skill_embeddings_cache = None
        self._write_skill_md(clean_skill)
        self._skills_fingerprint = self._compute_skills_fingerprint()
        logger.info("[SkillManager] added skill: %s", name)
        return True

    def add_skills(self, new_skills: list[dict], category: str = "general") -> int:
        """Add multiple skills; returns count actually added.

        Increments ``self.generation`` when at least one skill is successfully
        added, signalling callers that the local skill library changed.
        """
        added = 0
        for skill in new_skills:
            if "category" not in skill:
                skill = {**skill, "category": category}
            if self.add_skill(skill):
                added += 1
        if added > 0:
            self.generation += 1
        return added

    @staticmethod
    def _format_frontmatter(skill: dict) -> str:
        """Serialize skill frontmatter as YAML, OpenClaw-compatible.

        Produces output like::

            name: my-skill
            description: "Rich description with: colons and special chars."
            homepage: https://example.com
            metadata:
              {
                "openclaw": { "emoji": "🔧" },
                "skillclaw": { "category": "coding" }
              }

        Extra frontmatter fields (``homepage``, ``user-invocable``, etc.)
        are written between ``description`` and ``metadata`` to match the
        OpenClaw SKILL.md convention.
        """
        lines: list[str] = []
        name = skill.get("name", "unknown")
        description = skill.get("description", "")
        metadata = skill.get("metadata")
        extra_fm = skill.get("_extra_frontmatter", {})

        lines.append(f"name: {name}")

        needs_quoting = any(c in description for c in ":{}[],\"'#&*!|>%@`\n")
        if needs_quoting:
            escaped = description.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            lines.append(f'description: "{escaped}"')
        else:
            lines.append(f"description: {description}")

        # Write extra OpenClaw frontmatter fields (homepage, user-invocable, etc.)
        for key, value in extra_fm.items():
            if key.startswith("_"):
                continue
            dumped = yaml.dump(
                {key: value},
                default_flow_style=False,
                allow_unicode=True,
                width=10000,
            ).strip()
            lines.append(dumped)

        if metadata and isinstance(metadata, dict):
            json_str = json.dumps(metadata, ensure_ascii=False, indent=2)
            indented = "\n".join("  " + ln for ln in json_str.splitlines())
            lines.append(f"metadata:\n{indented}")

        return "\n".join(lines)

    def _write_skill_md(self, skill: dict) -> None:
        """Persist a single skill to its SKILL.md file (AgentSkills / OpenClaw
        compatible format).

        Category is stored in ``metadata.skillclaw.category`` (not as a bare
        top-level frontmatter field) so the output is fully OpenClaw-compatible.
        Extra frontmatter fields (e.g. ``homepage``) that were parsed from an
        existing SKILL.md are preserved in the output.
        """
        name = skill.get("name", "unknown")
        skill_dir = self._skill_dir_path(skill)
        canonical = os.path.realpath(skill_dir)
        if not canonical.startswith(os.path.realpath(self._skills_dir) + os.sep):
            logger.warning("[SkillManager] blocked path traversal in skill name: %s", name)
            return
        os.makedirs(skill_dir, exist_ok=True)
        filepath = self._skill_md_path(skill)

        category = skill.get("category", "general")
        content = skill.get("content", "")
        metadata = dict(skill.get("metadata") or {})

        if category and category != "general":
            metadata.setdefault("skillclaw", {})["category"] = category

        write_skill: Dict[str, Any] = {
            "name": name,
            "description": skill.get("description", ""),
        }
        extra_fm = skill.get("_extra_frontmatter")
        if extra_fm:
            write_skill["_extra_frontmatter"] = extra_fm
        if metadata:
            write_skill["metadata"] = metadata

        fm_text = self._format_frontmatter(write_skill)
        text = f"---\n{fm_text}\n---\n\n{content}\n"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            skill["file_path"] = os.path.realpath(filepath)
            logger.info("[SkillManager] wrote skill file: %s", filepath)
        except OSError as e:
            logger.warning("[SkillManager] could not write %s: %s", filepath, e)

    def save(self, path: Optional[str] = None) -> None:
        """
        Persist all in-memory skills back to .md files.

        ``path`` is ignored (kept for backward compatibility); files are always
        written to skills_dir.
        """
        all_skills = list(self.skills.get("all_skills", []))
        for skill in all_skills:
            self._write_skill_md(skill)
        self._skills_fingerprint = self._compute_skills_fingerprint()
        logger.info("[SkillManager] saved %d skills to %s", len(all_skills), self._skills_dir)

    def _get_all_skill_names(self) -> set:
        return {str(s.get("name")) for s in self.skills.get("all_skills", []) if s.get("name")}

    def _category_counts(self) -> Counter:
        return Counter(str(s.get("category") or "general") for s in self.skills.get("all_skills", []))

    def get_skill_count(self) -> dict:
        return {
            "total": len(self.skills.get("all_skills", [])),
            "by_category": dict(self._category_counts()),
        }
