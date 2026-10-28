"""Run the Studio typography race regressions with Node's standard runner."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required")
def test_queued_typography_edits_preserve_document_state() -> None:
    """Exercise asynchronous browser application state through public APIs."""
    suite = Path(__file__).with_name("typography.test.mjs")
    result = subprocess.run(
        ["node", "--experimental-default-type=module", "--test", str(suite)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
