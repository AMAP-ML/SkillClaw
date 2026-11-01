"""Regression tests for SkillClaw #107.

In --mock mode EvolveServer built its main bucket through the mock-aware
_build_bucket() but built the ValidationStore straight from the raw config,
whose storage_backend is "" in a pure mock run, so startup crashed with
``ValueError: Unsupported storage backend: ''``.
"""

from evolve_server.core.config import EvolveServerConfig
from evolve_server.engines.workflow import EvolveServer
from evolve_server.storage.mock_bucket import _DEFAULT_LOCAL_DIR
from skillclaw.object_store import LocalObjectStore


def _config(**overrides):
    config = EvolveServerConfig()
    # Server construction builds an OpenAI client; a key must be present.
    config.llm_api_key = "sk-test"
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _validation_store(server):
    return server._validation_store._bucket


def test_mock_mode_builds_local_validation_store_at_mock_root(tmp_path):
    server = EvolveServer(_config(), mock=True, mock_root=str(tmp_path))

    store = _validation_store(server)
    assert isinstance(store, LocalObjectStore)
    assert store._root == str(tmp_path)


def test_mock_mode_without_mock_root_uses_default_local_dir():
    server = EvolveServer(_config(), mock=True)

    store = _validation_store(server)
    assert isinstance(store, LocalObjectStore)
    assert store._root == str(_DEFAULT_LOCAL_DIR)


def test_mock_mode_ignores_configured_remote_backend(tmp_path):
    # _build_bucket ignores the configured backend in mock mode; the
    # validation store must follow the same rule.
    config = _config(
        storage_backend="s3",
        storage_bucket="some-bucket",
        storage_endpoint="https://s3.example.test",
    )
    server = EvolveServer(config, mock=True, mock_root=str(tmp_path))

    store = _validation_store(server)
    assert isinstance(store, LocalObjectStore)
    assert store._root == str(tmp_path)


def test_non_mock_mode_still_uses_configured_backend(tmp_path):
    config = _config(storage_backend="local", local_root=str(tmp_path))
    server = EvolveServer(config, mock=False)

    store = _validation_store(server)
    assert isinstance(store, LocalObjectStore)
    assert store._root == str(tmp_path)
