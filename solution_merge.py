import argparse
import json
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description="Merge json inside a folder")
parser.add_argument("--folder", type=str, help="folder containing json files")
parser.add_argument("--output", type=str, help="output file name")
args = parser.parse_args()

folder = Path(args.folder)
output = Path(args.output)

if not folder.exists():
    raise Exception("Folder does not exist")

merged = defaultdict(list)
for file in folder.iterdir():
    if file.suffix == ".json":
        with open(file, "r") as f:
            data = json.load(f)
            for k, v in data.items():
                merged[k].extend(v)

with open(output, "w") as f:
    json.dump(merged, f)
