"""Read-only comparison with the pre-rewrite Git baseline; no dialogue generation."""
import json
from pathlib import Path
import subprocess
import build_curated_dataset as builder


def main():
    manifest = builder.read(builder.MANIFEST)
    baseline = manifest["baseline_commit"]
    result = {}
    for split, filename in builder.FILES.items():
        path = f"data/{filename}.json"
        original = json.loads(subprocess.check_output(
            ["git", "show", f"{baseline}:{path}"], cwd=builder.ROOT).decode("utf-8-sig"))
        current = builder.read(builder.ROOT / path)
        if len(original) != len(current):
            raise ValueError(f"{split}: total slot count changed")
        ledger = {r["index"]: r for r in manifest["records"] if r["split"] == split}
        untouched = 0
        for index, (before, after) in enumerate(zip(original, current)):
            if index not in ledger:
                if before != after:
                    raise ValueError(f"{split}: untracked change at index {index}")
                untouched += 1
                continue
            row = ledger[index]
            if builder.digest(before) != row["original_sha256"]:
                raise ValueError(f"{split}: baseline hash mismatch at {index}")
            if builder.digest(after) != row["compiled_sha256"]:
                raise ValueError(f"{split}: compiled hash mismatch at {index}")
            if builder.npc_name(before) != builder.npc_name(after):
                raise ValueError(f"{split}: NPC identity changed without explicit migration at {index}")
        result[split] = dict(rewritten_verified=len(ledger), untouched_verified=untouched)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
