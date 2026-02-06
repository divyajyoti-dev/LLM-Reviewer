import json
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: Path | str) -> List[Dict]:
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

