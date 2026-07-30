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


def test_setup_wizard_minimax_preset_defaults_to_m3_and_global_region(monkeypatch, tmp_path):
    saved = {}

    class FakeConfigStore:
        config_file = tmp_path / "config.yaml"

        def exists(self):
            return False

        def load(self):
            return {}

        def save(self, data):
            saved.update(data)

    def fake_prompt_choice(msg, choices, default=""):
        if msg == "LLM provider":
            return "minimax"
        return default if default else choices[0]

    monkeypatch.setattr(setup_wizard, "ConfigStore", FakeConfigStore)
    monkeypatch.setattr(setup_wizard, "_prompt_choice", fake_prompt_choice)
    monkeypatch.setattr(setup_wizard, "_prompt", lambda msg, default="", hide=False: default)
    monkeypatch.setattr(setup_wizard, "_prompt_bool", lambda msg, default=False: default)
    monkeypatch.setattr(setup_wizard, "_prompt_int", lambda msg, default=0: default)

    SetupWizard().run()

    assert saved["llm"]["provider"] == "minimax"
    assert saved["llm"]["model_id"] == "MiniMax-M3"
    assert saved["llm"]["api_base"] == "https://api.minimax.io/v1"
    assert saved["llm"]["region"] == "global_en"


def test_setup_wizard_minimax_preset_china_region_selects_cn_endpoint(monkeypatch, tmp_path):
    saved = {}

    class FakeConfigStore:
        config_file = tmp_path / "config.yaml"

        def exists(self):
            return False

        def load(self):
            return {}

        def save(self, data):
            saved.update(data)

    def fake_prompt_choice(msg, choices, default=""):
        if msg == "LLM provider":
            return "minimax"
        if msg == "Region":
            return "cn_zh"
        return default if default else choices[0]

    monkeypatch.setattr(setup_wizard, "ConfigStore", FakeConfigStore)
    monkeypatch.setattr(setup_wizard, "_prompt_choice", fake_prompt_choice)
    monkeypatch.setattr(setup_wizard, "_prompt", lambda msg, default="", hide=False: default)
    monkeypatch.setattr(setup_wizard, "_prompt_bool", lambda msg, default=False: default)
    monkeypatch.setattr(setup_wizard, "_prompt_int", lambda msg, default=0: default)

    SetupWizard().run()

    assert saved["llm"]["provider"] == "minimax"
    assert saved["llm"]["api_base"] == "https://api.minimaxi.com/v1"
    assert saved["llm"]["region"] == "cn_zh"
