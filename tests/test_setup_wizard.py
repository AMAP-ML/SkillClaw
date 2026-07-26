from skillclaw import setup_wizard
from skillclaw.setup_wizard import SetupWizard


def test_setup_wizard_preserves_existing_llm_api_mode(monkeypatch, tmp_path):
    saved = {}

    class FakeConfigStore:
        config_file = tmp_path / "config.yaml"

        def exists(self):
            return True

        def load(self):
            return {
                "claw_type": "codex",
                "llm": {
                    "provider": "custom",
                    "model_id": "upstream-model",
                    "api_base": "http://upstream.test/v1",
                    "api_key": "upstream-key",
                    "api_mode": "responses",
                },
                "proxy": {"port": 30000, "served_model_name": "skillclaw-model"},
                "skills": {"enabled": False, "dir": str(tmp_path / "skills")},
                "prm": {"enabled": False},
                "sharing": {"enabled": False},
            }

        def save(self, data):
            saved.update(data)

    monkeypatch.setattr(setup_wizard, "ConfigStore", FakeConfigStore)
    monkeypatch.setattr(setup_wizard, "_prompt_choice", lambda msg, choices, default="": default)
    monkeypatch.setattr(setup_wizard, "_prompt", lambda msg, default="", hide=False: default)
    monkeypatch.setattr(setup_wizard, "_prompt_bool", lambda msg, default=False: default)
    monkeypatch.setattr(setup_wizard, "_prompt_int", lambda msg, default=0: default)

    SetupWizard().run()

    assert saved["llm"]["api_mode"] == "responses"


def test_setup_wizard_resets_api_mode_when_provider_changes(monkeypatch, tmp_path):
    saved = {}

    class FakeConfigStore:
        config_file = tmp_path / "config.yaml"

        def exists(self):
            return True

        def load(self):
            return {
                "claw_type": "hermes",
                "llm": {
                    "provider": "hermes-openai-codex",
                    "model_id": "gpt-5.6-sol",
                    "api_base": "",
                    "api_key": "",
                    "api_mode": "responses",
                },
                "proxy": {"port": 30000, "served_model_name": "skillclaw-model"},
                "skills": {"enabled": False, "dir": str(tmp_path / "skills")},
                "prm": {"enabled": False},
                "sharing": {"enabled": False},
            }

        def save(self, data):
            saved.update(data)

    def choose(message, choices, default=""):
        return "openrouter" if message == "LLM provider" else default

    monkeypatch.setattr(setup_wizard, "ConfigStore", FakeConfigStore)
    monkeypatch.setattr(setup_wizard, "_prompt_choice", choose)
    monkeypatch.setattr(setup_wizard, "_prompt", lambda msg, default="", hide=False: default)
    monkeypatch.setattr(setup_wizard, "_prompt_bool", lambda msg, default=False: default)
    monkeypatch.setattr(setup_wizard, "_prompt_int", lambda msg, default=0: default)

    SetupWizard().run()

    assert saved["llm"]["provider"] == "openrouter"
    assert saved["llm"]["api_mode"] == "chat"


def test_setup_wizard_configures_hermes_codex_oauth_without_api_key(monkeypatch, tmp_path):
    saved = {}

    class FakeConfigStore:
        config_file = tmp_path / "config.yaml"

        def exists(self):
            return False

        def load(self):
            return {}

        def save(self, data):
            saved.update(data)

    monkeypatch.setattr(setup_wizard, "ConfigStore", FakeConfigStore)
    monkeypatch.setattr(setup_wizard, "_validate_hermes_codex_oauth", lambda: None)
    monkeypatch.setattr(setup_wizard.secrets, "token_urlsafe", lambda _: "generated-proxy-key")

    def choose(message, choices, default=""):
        if message == "CLI agent to configure":
            return "hermes"
        if message == "LLM provider":
            return "hermes-openai-codex"
        return default

    monkeypatch.setattr(setup_wizard, "_prompt_choice", choose)
    monkeypatch.setattr(setup_wizard, "_prompt", lambda msg, default="", hide=False: default)
    monkeypatch.setattr(setup_wizard, "_prompt_bool", lambda msg, default=False: default)
    monkeypatch.setattr(setup_wizard, "_prompt_int", lambda msg, default=0: default)

    SetupWizard().run()

    assert saved["claw_type"] == "hermes"
    assert saved["llm"] == {
        "provider": "hermes-openai-codex",
        "model_id": "gpt-5.6-sol",
        "api_base": "",
        "api_key": "",
        "api_mode": "responses",
        "bedrock_region": "",
    }
    assert saved["proxy"]["host"] == "127.0.0.1"
    assert saved["proxy"]["api_key"] == "generated-proxy-key"
    assert saved["prm"] == {"enabled": False}


def test_validate_hermes_codex_oauth_reports_login_remediation(monkeypatch):
    def unavailable():
        raise RuntimeError("missing Hermes source")

    monkeypatch.setattr("skillclaw.hermes_codex.resolve_upstream", unavailable)

    try:
        setup_wizard._validate_hermes_codex_oauth()
    except RuntimeError as exc:
        assert "hermes auth add openai-codex" in str(exc)
    else:
        raise AssertionError("missing OAuth must fail setup preflight")


def test_setup_rejects_codex_oauth_for_non_hermes_adapter(monkeypatch, tmp_path):
    class FakeConfigStore:
        config_file = tmp_path / "config.yaml"

        def exists(self):
            return False

        def load(self):
            return {}

    monkeypatch.setattr(setup_wizard, "ConfigStore", FakeConfigStore)

    def choose(message, choices, default=""):
        return "codex" if message == "CLI agent to configure" else "hermes-openai-codex"

    monkeypatch.setattr(setup_wizard, "_prompt_choice", choose)

    try:
        SetupWizard().run()
    except ValueError as exc:
        assert "only supported with the Hermes CLI adapter" in str(exc)
    else:
        raise AssertionError("non-Hermes adapters must reject Hermes-owned Codex OAuth")


def test_setup_preserves_existing_keyless_prm_url(monkeypatch, tmp_path):
    saved = {}

    class FakeConfigStore:
        config_file = tmp_path / "config.yaml"

        def exists(self):
            return True

        def load(self):
            return {
                "claw_type": "hermes",
                "llm": {
                    "provider": "hermes-openai-codex",
                    "model_id": "gpt-5.6-sol",
                    "api_mode": "responses",
                },
                "proxy": {"host": "127.0.0.1", "port": 30001, "api_key": "local-key"},
                "skills": {"enabled": False, "dir": str(tmp_path / "skills")},
                "prm": {"enabled": True, "url": "http://local-prm.test/score"},
                "sharing": {"enabled": False},
            }

        def save(self, data):
            saved.update(data)

    monkeypatch.setattr(setup_wizard, "ConfigStore", FakeConfigStore)
    monkeypatch.setattr(setup_wizard, "_validate_hermes_codex_oauth", lambda: None)
    monkeypatch.setattr(setup_wizard, "_prompt_choice", lambda msg, choices, default="": default)
    monkeypatch.setattr(setup_wizard, "_prompt", lambda msg, default="", hide=False: default)
    monkeypatch.setattr(setup_wizard, "_prompt_bool", lambda msg, default=False: default)
    monkeypatch.setattr(setup_wizard, "_prompt_int", lambda msg, default=0: default)

    SetupWizard().run()

    assert saved["prm"]["enabled"] is True
    assert saved["prm"]["url"] == "http://local-prm.test/score"
    assert saved["prm"]["api_key"] == ""
