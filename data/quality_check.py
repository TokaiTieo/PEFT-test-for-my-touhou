"""训练数据质检脚本（《东方异变录》契约版）。

规则（与 README 一致）：
1. output 必须是可解析的 JSON，且包含约定的 14 个顶层字段
2. description 长度 50-200 字（游戏规则）
3. memory_updates / open_event / spellcard_result 必须符合游戏后端实际结构
4. 重复 output 只保留一条（按完整字符串去重）
5. 敏感词过滤
6. instruction 必须包含「当前NPC信息」与「玩家的动作和语言」段落

用法：python data/quality_check.py [data/npc_dialogue.json data/npc_eval.json]
只报告不修改；退出码 0 = 全部通过，1 = 存在不合格样本。
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REQUIRED_KEYS = [
    "description", "is_dead", "time_cost", "new_energy_state", "new_location",
    "relationship_update", "inventory_updates", "reputation_updates", "world_effects",
    "task_updates", "memory_updates", "open_event", "spellcard_result", "exit_dialogue",
]
SENSITIVE_WORDS = [
    # TODO: 按需补充敏感词表
]
DESC_MIN, DESC_MAX = 50, 200
KNOWLEDGE_TYPES = {"direct", "reported", "inferred", "system"}
TRUTH_STATUSES = {"accepted", "disputed", "superseded"}
LIST_FIELDS = ["inventory_updates", "reputation_updates", "world_effects", "task_updates"]


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_contract(obj):
    """Validate the training-facing dialogue schema against touhou/backend.

    The backend supplies defaults for many fields, but training always uses the
    complete 14-field shape so the small model learns one deterministic layout.
    """
    if not isinstance(obj, dict):
        return ["output 顶层必须是 JSON 对象"]
    problems = []
    missing = [key for key in REQUIRED_KEYS if key not in obj]
    if missing:
        problems.append(f"缺少顶层字段 {missing}")

    if not isinstance(obj.get("description"), str):
        problems.append("description 必须是字符串")
    if not isinstance(obj.get("is_dead"), bool):
        problems.append("is_dead 必须是布尔值")
    if "exit_dialogue" in obj and not isinstance(obj.get("exit_dialogue"), bool):
        problems.append("exit_dialogue 必须是布尔值")
    if not _is_number(obj.get("time_cost")) or not 0 <= obj.get("time_cost", -1) <= 12:
        problems.append("time_cost 必须是 0-12 的数字")
    if obj.get("new_energy_state") is not None and not isinstance(obj.get("new_energy_state"), str):
        problems.append("new_energy_state 必须是字符串或 null")
    if obj.get("new_location") is not None and not isinstance(obj.get("new_location"), dict):
        problems.append("new_location 必须是对象或 null")
    if obj.get("relationship_update") is not None and not isinstance(obj.get("relationship_update"), str):
        problems.append("relationship_update 必须是字符串或 null")
    for key in LIST_FIELDS:
        if not isinstance(obj.get(key), list):
            problems.append(f"{key} 必须是数组")

    memories = obj.get("memory_updates")
    if not isinstance(memories, list):
        problems.append("memory_updates 必须是数组")
    else:
        for index, memory in enumerate(memories):
            prefix = f"memory_updates[{index}]"
            if not isinstance(memory, dict):
                problems.append(f"{prefix} 必须是对象，不能是字符串")
                continue
            if not isinstance(memory.get("npc_name"), str) or not memory["npc_name"].strip():
                problems.append(f"{prefix}.npc_name 必须是非空字符串")
            if not isinstance(memory.get("summary"), str) or not memory["summary"].strip():
                problems.append(f"{prefix}.summary 必须是非空字符串")
            elif len(memory["summary"]) > 300:
                problems.append(f"{prefix}.summary 超过后端持久化上限 300 字")
            if "tags" in memory and (
                not isinstance(memory["tags"], list)
                or not all(isinstance(tag, str) for tag in memory["tags"])
            ):
                problems.append(f"{prefix}.tags 必须是字符串数组")
            if "importance" in memory and (
                not isinstance(memory["importance"], int)
                or isinstance(memory["importance"], bool)
                or not 1 <= memory["importance"] <= 10
            ):
                problems.append(f"{prefix}.importance 必须是 1-10 的整数")
            if "knowledge_type" in memory and memory["knowledge_type"] not in KNOWLEDGE_TYPES:
                problems.append(f"{prefix}.knowledge_type 取值无效")
            if "confidence" in memory and (
                not _is_number(memory["confidence"]) or not 0 <= memory["confidence"] <= 1
            ):
                problems.append(f"{prefix}.confidence 必须在 0-1 之间")
            if "truth_status" in memory and memory["truth_status"] not in TRUTH_STATUSES:
                problems.append(f"{prefix}.truth_status 取值无效")

    event = obj.get("open_event")
    if event is not None:
        if not isinstance(event, dict):
            problems.append("open_event 必须是对象或 null")
        else:
            if not (event.get("title") or event.get("description")):
                problems.append("open_event 至少要有非空 title 或 description")
            for key in ("title", "type", "scene", "description"):
                if not isinstance(event.get(key), str) or not event[key].strip():
                    problems.append(f"open_event.{key} 必须是非空字符串")
            if (
                not isinstance(event.get("hooks"), list)
                or not all(isinstance(hook, str) for hook in event["hooks"])
            ):
                problems.append("open_event.hooks 必须是字符串数组")

    battle = obj.get("spellcard_result")
    if battle is not None:
        if not isinstance(battle, dict):
            problems.append("spellcard_result 必须是对象或 null")
        else:
            for key in ("opponent", "spellcard_name", "outcome", "summary"):
                if not isinstance(battle.get(key), str) or not battle[key].strip():
                    problems.append(f"spellcard_result.{key} 必须是非空字符串")
            if "cost" not in battle:
                problems.append("spellcard_result 缺少 cost")

    for index, item in enumerate(obj.get("inventory_updates") or []):
        prefix = f"inventory_updates[{index}]"
        if not isinstance(item, dict):
            problems.append(f"{prefix} 必须是对象")
        elif item.get("action") not in {"add", "remove", "use"} or not item.get("name"):
            problems.append(f"{prefix} 需要有效 action 与 name")
        elif not isinstance(item.get("quantity", 1), int) or not 1 <= item.get("quantity", 1) <= 999:
            problems.append(f"{prefix}.quantity 必须是 1-999 的整数")

    for index, item in enumerate(obj.get("reputation_updates") or []):
        prefix = f"reputation_updates[{index}]"
        if not isinstance(item, dict) or not item.get("faction"):
            problems.append(f"{prefix} 需要非空 faction")
        elif not _is_number(item.get("delta")) or not -12 <= item["delta"] <= 12:
            problems.append(f"{prefix}.delta 必须在 -12 到 12 之间")

    for index, item in enumerate(obj.get("world_effects") or []):
        prefix = f"world_effects[{index}]"
        if not isinstance(item, dict) or item.get("kind") not in {"location", "rumor", "flag"}:
            problems.append(f"{prefix}.kind 取值无效")
        elif not isinstance(item.get("effect"), str) or not item["effect"].strip():
            problems.append(f"{prefix}.effect 必须是非空字符串")

    for index, item in enumerate(obj.get("task_updates") or []):
        prefix = f"task_updates[{index}]"
        if not isinstance(item, dict) or item.get("action") not in {"add", "update", "complete", "update_priority"}:
            problems.append(f"{prefix}.action 取值无效")
        elif item.get("action") != "add" and not item.get("task_id"):
            problems.append(f"{prefix} 非 add 操作必须提供 task_id")

    return problems


def check(samples):
    errors = Counter()
    seen_outputs = set()
    for i, s in enumerate(samples):
        tag = f"#{i}"
        if not isinstance(s, dict):
            errors[f"{tag} 样本必须是对象"] += 1
            continue
        ins, out = s.get("instruction", ""), s.get("output", "")
        if set(s) != {"instruction", "input", "output"}:
            errors[f"{tag} 外层必须且只能包含 instruction/input/output"] += 1
        if not all(isinstance(s.get(key), str) for key in ("instruction", "input", "output")):
            errors[f"{tag} instruction/input/output 必须都是字符串"] += 1
            continue

        # 规则 5：instruction 结构
        for seg in ("当前NPC信息", "玩家的动作和语言"):
            if seg not in ins:
                errors[f"{tag} instruction 缺少「{seg}」段落"] += 1

        # 规则 1：JSON 契约
        try:
            obj = json.loads(out)
        except json.JSONDecodeError:
            errors[f"{tag} output 不是合法 JSON"] += 1
            continue
        for problem in validate_contract(obj):
            errors[f"{tag} {problem}"] += 1
        if not isinstance(obj, dict):
            continue

        # 规则 2：description 长度
        desc = obj.get("description", "")
        if not (DESC_MIN <= len(desc) <= DESC_MAX):
            errors[f"{tag} description 长度 {len(desc)} 字，超出 {DESC_MIN}-{DESC_MAX}"] += 1

        # 规则 3：重复
        if out in seen_outputs:
            errors[f"{tag} output 与之前样本完全重复"] += 1
        seen_outputs.add(out)

        # 规则 4：敏感词
        for w in SENSITIVE_WORDS:
            if re.search(re.escape(w), out):
                errors[f"{tag} 命中敏感词「{w}」"] += 1

    return errors, len(samples)


def fingerprint(sample, field):
    value = sample["instruction"] + "\n" + sample.get("input", "") if field == "prompt" else sample["output"]
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="检查 NPC 数据契约、重复与跨集合泄漏")
    parser.add_argument("paths", nargs="*", default=["data/npc_dialogue.json"])
    args = parser.parse_args()
    loaded = []
    failed = False
    for raw_path in args.paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as handle:
            samples = json.load(handle)
        if not isinstance(samples, list):
            print(f"{path}: 顶层必须是数组")
            failed = True
            continue
        errors, total = check(samples)
        expected = 2000 if path.name == "npc_dialogue.json" else 200 if path.name == "npc_eval.json" else None
        if expected is not None and total != expected:
            errors[f"条数应为 {expected}，实际 {total}"] += 1
        print(f"{path}: 共检查 {total} 条样本")
        if errors:
            failed = True
            print(f"发现 {sum(errors.values())} 处问题：")
            for msg, n in errors.items():
                print(f"  [{n}x] {msg}")
        else:
            print("全部通过 ✓")
        loaded.append((path, samples))

    if len(loaded) > 1:
        for i, (left_path, left) in enumerate(loaded):
            for right_path, right in loaded[i + 1:]:
                for field in ("prompt", "output"):
                    overlap = {fingerprint(x, field) for x in left} & {fingerprint(x, field) for x in right}
                    print(f"{left_path.name} ↔ {right_path.name}: exact {field} 重叠 {len(overlap)}")
                    failed |= bool(overlap)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
