"""Unit tests for configuration defaults and settings."""

from pathlib import Path

from opendb_core.config import Settings


class TestSettingsDefaults:
    def test_default_backend(self) -> None:
        # The embedded backend is what `pip install open-db` ships and what
        # every quickstart assumes; defaulting to postgres pointed the
        # documented three-line install at a server nobody was told to run.
        s = Settings()
        assert s.backend == "sqlite"

    def test_default_max_file_size(self) -> None:
        s = Settings()
        assert s.max_file_size == 100 * 1024 * 1024  # 100 MB

    def test_default_port(self) -> None:
        s = Settings()
        assert s.port == 8000

    def test_default_host(self) -> None:
        # Loopback by default. Binding 0.0.0.0 with auth unenforced exposed the
        # whole workspace to the local network.
        s = Settings()
        assert s.host == "127.0.0.1"

    def test_default_ocr_enabled(self) -> None:
        s = Settings()
        assert s.ocr_enabled is True

    def test_default_ocr_languages(self) -> None:
        s = Settings()
        assert "eng" in s.ocr_languages

    def test_default_memory_decay(self) -> None:
        s = Settings()
        assert s.memory_decay_halflife_days == 30.0

    def test_default_auth_key_empty(self) -> None:
        s = Settings()
        assert s.auth_api_key == ""

    def test_default_watch_max(self) -> None:
        s = Settings()
        assert s.watch_max_watchers == 10

    def test_default_index_max_concurrent(self) -> None:
        s = Settings()
        assert s.index_max_concurrent == 4

    def test_default_cors_origins(self) -> None:
        # No cross-origin by default: with "*" any web page the developer
        # visited could script the local OpenDB server.
        s = Settings()
        assert s.cors_origins == []

    def test_vision_egress_is_opt_in(self) -> None:
        # Enabling vision POSTs indexed image bytes to a third-party API.
        s = Settings()
        assert s.vision_enabled is False

    def test_file_storage_path_is_path(self) -> None:
        s = Settings()
        assert isinstance(s.file_storage_path, Path)

    def test_opendb_dir_is_path(self) -> None:
        s = Settings()
        assert isinstance(s.opendb_dir, Path)
