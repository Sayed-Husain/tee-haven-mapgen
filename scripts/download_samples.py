"""Download sample .map files from OpenGores for testing."""

import urllib.request
from pathlib import Path

REPO_BASE = "https://raw.githubusercontent.com/teemods/opengores-maps/main/maps"

SAMPLES = [
    "Simpler",      # easy — good for initial testing
    "Encore",        # medium
    "Bl0odDens5",    # hard
]

def main():
    out_dir = Path(__file__).resolve().parent.parent / "maps" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in SAMPLES:
        dest = out_dir / f"{name}.map"
        if dest.exists():
            print(f"  skip  {name}.map (already exists)")
            continue

        url = f"{REPO_BASE}/{name}.map"
        print(f"  fetch  {name}.map ... ", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            size_kb = dest.stat().st_size / 1024
            print(f"ok ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    main()
