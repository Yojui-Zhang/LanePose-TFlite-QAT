from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    scripts = sorted(p for p in root.glob("verify_*.py"))
    if not scripts:
        raise FileNotFoundError("No verify_*.py scripts found under verify/.")

    for script in scripts:
        print(f"== {script.name} ==")
        result = subprocess.run([sys.executable, str(script)], check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        print("OK")

    print("All verify scripts passed.")


if __name__ == "__main__":
    main()
