# paths.py - project folder paths. Owner: Person C.
# Everyone imports paths from here; never type full paths like C:/Users/... .

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root folder
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


if __name__ == "__main__":
    print("ROOT      :", ROOT)
    print("DATA_DIR  :", DATA_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
