from pathlib import Path
import subprocess
import sys


def test_universal_spec_validator() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "validate_universal_spec.py")],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Universal interactive spec valid" in result.stdout
