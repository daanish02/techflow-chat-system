"""Run all tests and provide summary."""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run pytest and display results."""
    print("=" * 60)
    print("Running TechFlow Chat System Tests")
    print("=" * 60)
    print()

    result = subprocess.run(
        ["uv", "run", "pytest", "tests/", "-v", "--tb=short"],
        cwd=project_root,
        capture_output=False,
    )

    print()
    print("=" * 60)

    if result.returncode == 0:
        print("All tests passed!")
    else:
        print("Some tests failed")

    print("=" * 60)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
