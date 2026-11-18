from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

from evolve_server.core import config


def test_load_dotenv_from_evolve_server_package_root(monkeypatch, tmp_path) -> None:
    """Verify the documented package-root .env file is selected.

    @param monkeypatch Pytest fixture for restoring module state.
    @param tmp_path Pytest fixture for isolated filesystem paths.
    @returns None
    """
    env_path = tmp_path / "evolve_server" / ".env"
    env_path.parent.mkdir()
    env_path.touch()
    config_file = env_path.parent / "core" / "config.py"
    monkeypatch.setattr(config, "__file__", str(config_file))

    dotenv_module = ModuleType("dotenv")
    load_dotenv = Mock()
    dotenv_module.load_dotenv = load_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)

    config._load_dotenv()

    load_dotenv.assert_called_once_with(env_path, override=False)
