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
`skill_manage` calls, so the evolve server groups sessions by the skills they actually used
(otherwise every session lands in the no-skill bucket and existing skills are never improved).
All turns of one Hermes session go into one record; a session is published to the queue when it
is idle (default 30 min), when Hermes finalizes/resets it, or after 40 turns.

## Install

```bash
cp -r integrations/hermes_plugin/skillclaw_capture ~/.hermes/plugins/skillclaw-capture
# per profile: ~/.hermes/profiles/<name>/plugins/skillclaw-capture
```

Add to the Hermes `.env` (per profile if you use several):

```bash
SKILLCLAW_CAPTURE_ROOT=~/.skillclaw/local-share   # same as the server's local root
SKILLCLAW_GROUP_ID=my-group
# optional
SKILLCLAW_USER_ALIAS=alice
SKILLCLAW_CAPTURE_IDLE_SECONDS=1800
```

Then enable it: `hermes plugins enable skillclaw-capture`.

Run the evolve server against the same root, with its own LLM endpoint:

```bash
skillclaw-evolve-server --storage-backend local --local-root ~/.skillclaw/local-share --group-id my-group
```

Do **not** also run the proxy integration for the same Hermes profile, or turns are recorded twice.

## What is stored

User message and final assistant text of each turn (with secret redaction), model name, tool
**names** and skill names. System prompts, earlier history, tool arguments and tool results are
not stored. Redaction removes credentials but does not anonymize text — keep private and work
profiles in different groups.
