"""Tests for Streamlit replay viewer loading and launch behavior."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from bridge_ai.ui import streamlit_app


class ReplayLoaderTests(unittest.TestCase):
    def test_load_replays_finds_nested_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a").mkdir(parents=True, exist_ok=True)
            (root / "b" / "c").mkdir(parents=True, exist_ok=True)
            (root / "a" / "latest.json").write_text(json.dumps([{"step": 1}]), encoding="utf-8")
            (root / "b" / "c" / "episode.json").write_text(json.dumps([{"step": 2}]), encoding="utf-8")

            loaded = streamlit_app._load_replays(str(root))

        self.assertEqual(sorted(loaded.keys()), ["a/latest.json", "b/c/episode.json"])


class LauncherBehaviorTests(unittest.TestCase):
    def test_main_launches_streamlit_when_not_in_streamlit_context(self) -> None:
        with mock.patch.object(streamlit_app, "st", object()), \
            mock.patch.object(streamlit_app, "_is_running_in_streamlit", return_value=False), \
            mock.patch.object(streamlit_app, "_launch_streamlit") as launch_mock, \
            mock.patch.object(streamlit_app, "run_ui") as run_ui_mock, \
            mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(streamlit_app._LAUNCH_GUARD_ENV, None)
            streamlit_app.main()

        launch_mock.assert_called_once_with(str(Path(streamlit_app.__file__).resolve()))
        run_ui_mock.assert_not_called()

    def test_main_runs_ui_in_streamlit_context(self) -> None:
        with mock.patch.object(streamlit_app, "st", object()), \
            mock.patch.object(streamlit_app, "_is_running_in_streamlit", return_value=True), \
            mock.patch.object(streamlit_app, "_launch_streamlit") as launch_mock, \
            mock.patch.object(streamlit_app, "run_ui") as run_ui_mock:
            streamlit_app.main()

        launch_mock.assert_not_called()
        run_ui_mock.assert_called_once()

    def test_main_runs_ui_when_launch_guard_is_set(self) -> None:
        with mock.patch.object(streamlit_app, "st", object()), \
            mock.patch.object(streamlit_app, "_is_running_in_streamlit", return_value=False), \
            mock.patch.object(streamlit_app, "_launch_streamlit") as launch_mock, \
            mock.patch.object(streamlit_app, "run_ui") as run_ui_mock, \
            mock.patch.dict(os.environ, {streamlit_app._LAUNCH_GUARD_ENV: "1"}, clear=False):
            streamlit_app.main()

        launch_mock.assert_not_called()
        run_ui_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
