"""训练数据质检脚本（《东方异变录》契约版）。

规则（与 README 一致）：
1. output 必须是可解析的 JSON，且含必填字段 description / is_dead
2. description 长度 50-200 字（游戏规则）
3. 重复 output 只保留一条（按完整字符串去重）
4. 敏感词过滤
5. instruction 必须包含「当前NPC信息」与「玩家的动作和语言」段落

用法：python data/quality_check.py [data/npc_dialogue.json]
只报告不修改；退出码 0 = 全部通过，1 = 存在不合格样本。
"""

import json
import re
import sys
from collections import Counter

REQUIRED_KEYS = ["description", "is_dead"]
SENSITIVE_WORDS = [
    # TODO: 按需补充敏感词表
]
DESC_MIN, DESC_MAX = 50, 200


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
        missing = [k for k in REQUIRED_KEYS if k not in obj]
        if missing:
            errors[f"{tag} 缺少必填字段 {missing}"] += 1

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
