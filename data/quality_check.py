"""训练数据质检脚本（《东方异变录》契约版）。

规则（与 README 一致）：
1. output 必须是可解析的 JSON，且包含约定的 14 个顶层字段
2. description 长度 50-200 字（游戏规则）
3. memory_updates / open_event / spellcard_result 必须符合游戏后端实际结构
4. 重复 output 只保留一条（按完整字符串去重）
5. 敏感词过滤
6. instruction 必须包含「当前NPC信息」与「玩家的动作和语言」段落

用法：python data/quality_check.py [data/npc_dialogue.json]
只报告不修改；退出码 0 = 全部通过，1 = 存在不合格样本。
"""

import json
import re
import sys
from collections import Counter

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

    return problems


def check(samples):
    errors = Counter()
    seen_outputs = set()
    for i, s in enumerate(samples):
        tag = f"#{i}"
        ins, out = s.get("instruction", ""), s.get("output", "")

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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/npc_dialogue.json"
    with open(path, encoding="utf-8") as f:
        samples = json.load(f)
    errors, total = check(samples)
    print(f"共检查 {total} 条样本")
    if errors:
        print(f"发现 {sum(errors.values())} 处问题：")
        for msg, n in errors.items():
            print(f"  [{n}x] {msg}")
        sys.exit(1)
    print("全部通过 ✓")


if __name__ == "__main__":
    main()
