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
