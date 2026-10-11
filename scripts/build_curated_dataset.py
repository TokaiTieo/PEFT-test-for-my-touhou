"""Assemble directly authored records; never generate or paraphrase dialogue.

Default is a read-only consistency check. --write replaces only indexed NPC
slots after validating the entire batch. Requires no API or ML dependencies.
"""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
from quality_check import check

FILES = {"train": "npc_dialogue", "eval": "npc_eval"}
MANIFEST = ROOT / "data/canon_review_manifest.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def serialized(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def npc_name(sample):
    names = [line.split("：", 1)[1].strip() for line in sample["instruction"].splitlines()
             if line.startswith("姓名：")]
    if len(names) != 1:
        raise ValueError("Expected exactly one NPC name in instruction")
    return names[0]


def load_authored():
    records, seen = [], set()
    for path in sorted((ROOT / "data/curated").glob("batch_*.txt")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line or line.startswith("#"):
                continue
            fields = line.split("|")
            if len(fields) != 7:
                raise ValueError(f"{path.name}:{line_number}: expected 7 fields")
            split, name, case, context, player, description, delta = fields
            record_id = f"{split}/{name}/{case}"
            if record_id in seen or split not in FILES:
                raise ValueError(f"Duplicate or invalid record: {record_id}")
            seen.add(record_id)
            records.append(dict(id=record_id, split=split, npc=name, case=case,
                                context=context, player=player, description=description,
                                delta=json.loads(delta), source_file=str(path.relative_to(ROOT)).replace("\\", "/"),
                                source_line=line_number))
    if not records:
        raise ValueError("No directly authored records")
    return records


def assemble(record, profile):
    # Only fixed schema defaults are added here. Every description and every
    # nonempty state update comes verbatim from the authored source file.
    answer = dict(description=record["description"], is_dead=False, time_cost=0,
                  new_energy_state=None, new_location=None, relationship_update=None,
                  inventory_updates=[], reputation_updates=[], world_effects=[],
                  task_updates=[], memory_updates=[], open_event=None,
                  spellcard_result=None, exit_dialogue=False)
    if not isinstance(record["delta"], dict) or set(record["delta"]) - (set(answer) - {"description"}):
        raise ValueError(f"Unsupported state delta: {record['id']}")
    answer.update(record["delta"])
    instruction = (
        "你是《东方异变录》的NPC。只输出一个JSON对象，不要代码围栏。\n\n"
        "## 当前NPC信息\n"
        f"姓名：{record['npc']}\n设定：{profile['canon']}\n"
        f"表达方向：{profile['voice']}\n避免误写：{'；'.join(profile['avoid'])}。\n\n"
        "## 玩家与场景状态\n"
        f"{record['context']}\n"
        "这是当前游戏的虚构情境，不是官方剧情。只采用明确给出的事实，"
        "未说明的关系、物品和任务不能自行当成已经存在。\n\n"
        "## 玩家的动作和语言\n"
        f"玩家：{record['player']}\n\n"
        "## 输出规则\n"
        "description为50–200字的场景回应，不能把人物简介当对白复述。"
        "只记录本回合确实发生的新状态；回忆旧礼物、既有关系或已结算比试不重复入账。"
        "不要替玩家同意约定或编造后端胜负；当前局若给出裁定，必须遵守。\n"
        "输出字段：description、is_dead、time_cost、new_energy_state、new_location、"
        "relationship_update、inventory_updates、reputation_updates、world_effects、"
        "task_updates、memory_updates、open_event、spellcard_result、exit_dialogue。"
        "没有变化的数组为[]，可选对象及relationship_update为null；is_dead和exit_dialogue默认false，"
        "未明确消耗的time_cost为0。memory_updates用对象数组，每项包含npc_name、summary，"
        "可含tags、importance、emotion；open_event包含title、type、scene、description、hooks；"
        "spellcard_result包含opponent、spellcard_name、outcome、summary、cost。"
    )
    return {"instruction": instruction, "input": "",
            "output": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))}


def audit(records, reviewed, canonical_by_npc=None):
    canonical_by_npc = canonical_by_npc or {}
    for split, samples in reviewed.items():
        errors, _ = check(samples)
        if errors:
            raise ValueError(f"{split}: {dict(errors)}")
    seen = {"prompt": set(), "description": set()}
    text_pairs = []
    for record in records:
        pair = record["context"] + "\n" + record["player"]
        for field, value in (("prompt", pair), ("description", record["description"])):
            if value in seen[field]:
                raise ValueError(f"Duplicate {field}: {record['id']}")
            seen[field].add(value)
        if (record["case"].startswith("old_") or record["case"] in {
                "review_battle", "past_gift", "old_promise", "relationship_stable"}) and record["delta"]:
            raise ValueError(f"Recall-only case has new state: {record['id']}")
        text_pairs.append((record, record["description"]))
    # Lexical diagnostic, NOT proof of independent meanings or canon accuracy.
    largest = (0.0, "", "")
    cross = (0.0, "", "")
    for index, (left, a) in enumerate(text_pairs):
        for right, b in text_pairs[index + 1:]:
            if canonical_by_npc.get(left["npc"], left["npc"]) != canonical_by_npc.get(right["npc"], right["npc"]):
                continue
            ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
            match = (ratio, left["id"], right["id"])
            largest = max(largest, match)
            if left["split"] != right["split"]:
                cross = max(cross, match)
    if largest[0] >= 0.8:
        raise ValueError(f"Near-duplicate description requires revision: {largest}")
    def result(match):
        return dict(ratio=round(match[0], 4), left=match[1], right=match[2])
    state_counts = {}
    for split, samples in reviewed.items():
        state_counts[split] = dict(Counter(key for s in samples for key, value in json.loads(s["output"]).items()
                                          if key not in {"description", "time_cost"} and value))
    return dict(exact_description_duplicates=0, exact_context_player_duplicates=0,
                similarity_identity="canonical source title (aliases grouped)",
                same_npc_max_sequence_similarity=result(largest),
                same_npc_cross_split_max_sequence_similarity=result(cross),
                state_positive_counts=state_counts,
                limits="字符串相似度不是语义去重、人设正确率或真实模型评测；仍需人工验收。")


def build():
    records = load_authored()
    profiles = {}
    for path in sorted((ROOT / "data").glob("canon_profiles*.json")):
        for profile in read(path)["profiles"]:
            if profile["name"] in profiles:
                raise ValueError(f"Duplicate NPC profile: {profile['name']}")
            profiles[profile["name"]] = profile
    previous = read(MANIFEST) if MANIFEST.exists() else {}
    prior_records = {(r["split"], r["index"]): r for r in previous.get("records", [])}
    sources = previous.get("sources", {})
    for name in {r["npc"] for r in records}:
        title = profiles[name]["source_title"]
        if title not in sources:
            cached = read(ROOT / ".cache/canon" / (title.replace("/", "_") + ".json"))
            sources[title] = {key: cached[key] for key in ("title", "revision", "url", "sha256")}
    groups = defaultdict(list)
    for record in records:
        groups[record["split"], record["npc"]].append(record)
    outputs, reviewed, ledger, progress = {}, {}, [], {}
    for split, filename in FILES.items():
        full = read(ROOT / "data" / (filename + ".json"))
        slots = defaultdict(list)
        for index, sample in enumerate(full):
            slots[npc_name(sample)].append(index)
        reviewed[split] = []
        covered = set()
        for (group_split, name), authored in groups.items():
            if group_split != split:
                continue
            if len(authored) > len(slots[name]):
                raise ValueError(f"{split}/{name}: {len(authored)} records vs {len(slots[name])} slots")
            for index, record in zip(slots[name], authored):
                old = prior_records.get((split, index))
                if old and old["id"] != record["id"]:
                    raise ValueError(f"Previously authored record moved at {split} index {index}; append without reordering")
                if old and digest(full[index]) != old["compiled_sha256"]:
                    raise ValueError(f"Dataset independently edited at {split} index {index}; reconcile first")
                original_hash = old["original_sha256"] if old else digest(full[index])
                sample = assemble(record, profiles[name])
                ledger.append(dict(id=record["id"], split=split, index=index, npc=name,
                                   source_file=record["source_file"], source_line=record["source_line"],
                                   source_title=profiles[name]["source_title"],
                                   status="assistant_rewritten_awaiting_human_review",
                                   original_sha256=original_hash, compiled_sha256=digest(sample)))
                full[index] = sample
                reviewed[split].append(sample)
                covered.add(index)
        outputs[ROOT / "data" / (filename + ".json")] = full
        outputs[ROOT / "data" / (filename + "_reviewed.json")] = reviewed[split]
        progress[split] = dict(total=len(full), rewritten=len(covered), pending=len(full) - len(covered),
                               pending_by_npc=dict(sorted(Counter(npc_name(sample) for index, sample in enumerate(full)
                                                                if index not in covered).items())))
    old_slots = {(r["split"], r["index"]) for r in previous.get("records", [])}
    if old_slots - {(r["split"], r["index"]) for r in ledger}:
        raise ValueError("Previously authored records removed; reconcile manifest before rebuilding")
    diagnostics = audit(records, reviewed, {name: p["source_title"] for name, p in profiles.items()})
    mapping_path = ROOT / "data/legacy_identity_mapping.json"
    legacy_mapping = read(mapping_path) if mapping_path.exists() else {"mappings": []}
    inferred_names = {row["legacy_label"] for row in legacy_mapping["mappings"]}
    inferred_count = sum(record["npc"] in inferred_names for record in records)
    complete = all(p["pending"] == 0 for p in progress.values())
    manifest = dict(schema_version=1, status="fully_rewritten_awaiting_human_review" if complete else "partial_rewrite_not_full_training_ready",
                    authoring="User-authorized collaborating assistants directly wrote complete scenes, prompts and responses; primary assistant reviewed samples and coordinated corrections; no paid model API; no response templates.",
                    human_review_completed=False, baseline_commit="0542f51c31313664934b18153527497d354f5b72",
                    scope=f"{len({r['npc'] for r in records})} NPC labels; fictional game scenarios, not official dialogue or recovered save transcripts",
                    contract_note="Legacy 14-field training shape retained; current DialogueTurnResult ignores is_dead/new_location.",
                    identity_resolution=dict(project_id_inferred_records=inferred_count,
                        other_source_backed_records=len(records) - inferred_count,
                        mapping_file="data/legacy_identity_mapping.json",
                        caveat="Legacy labels are not official aliases; ID-based mappings await user confirmation."),
                    sources=sources, progress=progress, diagnostics=diagnostics, records=ledger)
    outputs[MANIFEST] = manifest
    return outputs, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write validated assemblies to data/")
    args = parser.parse_args()
    outputs, manifest = build()
    stale = []
    for path, value in outputs.items():
        content = serialized(value)
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            stale.append(str(path.relative_to(ROOT)))
            if args.write:
                path.write_text(content, encoding="utf-8", newline="\n")
    counts = {split: {key: progress[key] for key in ("total", "rewritten", "pending")}
              for split, progress in manifest["progress"].items()}
    print(json.dumps({"progress": counts, "diagnostics": manifest["diagnostics"],
                      "changed" if args.write else "stale": stale}, ensure_ascii=False, indent=2))
    if stale and not args.write:
        raise SystemExit("Assembly differs; review sources then run with --write.")


if __name__ == "__main__":
    main()
