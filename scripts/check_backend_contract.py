"""Read-only check with the actual sibling game's parser; no server or API calls."""
import argparse
import json
from pathlib import Path
import sys


def preserved(raw, parsed, path="output"):
    if isinstance(raw, dict):
        assert isinstance(parsed, dict), path
        for key, value in raw.items():
            assert key in parsed, f"Parser discarded {path}.{key}"
            preserved(value, parsed[key], f"{path}.{key}")
    elif isinstance(raw, list):
        assert isinstance(parsed, list) and len(raw) == len(parsed), path
        for index, (left, right) in enumerate(zip(raw, parsed)):
            preserved(left, right, f"{path}[{index}]")
    else:
        assert raw == parsed, f"Parser changed {path}: {raw!r} -> {parsed!r}"


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--touhou-root", type=Path, default=root.parent / "touhou")
    args = ap.parse_args()
    if not (args.touhou_root / "backend/services/ai_contracts.py").is_file():
        raise SystemExit("Missing game backend; provide --touhou-root /path/to/touhou")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.touhou_root.resolve()))
    from backend.services.ai_contracts import DialogueTurnResult, parse_turn_response
    count = 0
    for filename in ("npc_dialogue_reviewed.json", "npc_eval_reviewed.json"):
        samples = json.loads((root / "data" / filename).read_text(encoding="utf-8"))
        for index, sample in enumerate(samples):
            raw = json.loads(sample["output"])
            parsed = parse_turn_response(sample["output"], DialogueTurnResult)
            assert parsed.get("contract_valid"), f"{filename}[{index}]: {parsed.get('contract_error')}"
            # Known legacy training fields are not dialogue-contract fields.
            assert raw.pop("is_dead") is False
            assert raw.pop("new_location") is None
            preserved(raw, parsed, f"{filename}[{index}]")
            count += 1
    print(f"Backend parsed {count} records; all non-legacy values preserved.")
    print("Legacy is_dead=false/new_location=null intentionally ignored by DialogueTurnResult.")


if __name__ == "__main__":
    main()
