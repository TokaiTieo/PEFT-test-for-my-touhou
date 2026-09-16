"""make_review_pack.py：从 generations.json 抽 30 条生成人工抽读包（Markdown）。

预筛标记（供人工重点复核，非定论）：
- [串味?] description 里出现指令主 NPC 以外的角色名
- [破墙?] description 里出现 JSON/字段/系统/大括号等元叙述
- [长度] description 不在 50-200 字

用法：
    python eval/make_review_pack.py eval/results/1.5b-ck564/generations.json \
        --npc-index npc_index.json --out review.md --n 30 --seed 20260902
"""
import argparse
import json
import random
import re
from pathlib import Path


def load_npc_names(path):
    d = json.load(open(path, encoding="utf-8-sig"))
    names = {n["name"] for n in d["npcs"]}
    names |= set(d.get("aliases", {}).keys())
    names |= set(d.get("aliases", {}).values())
    return sorted(names, key=len, reverse=True)  # 长名优先匹配


def extract_description(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


def main_npc(instruction, names):
    best, best_pos = None, len(instruction) + 1
    for n in names:
        p = instruction.find(n)
        if 0 <= p < best_pos:
            best, best_pos = n, p
    return best or "（未识别）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generations")
    ap.add_argument("--npc-index", required=True)
    ap.add_argument("--out", default="review_pack.md")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()

    entries = json.load(open(args.generations, encoding="utf-8"))
    names = load_npc_names(args.npc_index)
    idxs = sorted(random.Random(args.seed).sample(range(len(entries)), args.n))

    lines = [
        "# 人工抽读包（微调后输出，重点查人设串味）",
        "",
        f"- 来源：{args.generations}（共 {len(entries)} 条）",
        f"- 抽样：seed={args.seed} 随机 {args.n} 条",
        "- 读法：看 description 像不像这个角色在说话；有没有别的角色的口吻/名字乱入；",
        "  有没有跳出角色谈论 JSON、字段、系统（破墙）；预筛标记仅供参考，以人读为准。",
        "",
    ]
    flag_count = 0
    for rank, i in enumerate(idxs, 1):
        e = entries[i]
        npc = main_npc(e["instruction"], names)
        obj = extract_description(e["after"])
        desc = (obj or {}).get("description", e["after"][:200])
        flags = []
        if obj is not None:
            others = [n for n in names if n != npc and n in desc]
            if others:
                flags.append(f"[串味?] 出现 {','.join(others[:3])}")
            if re.search(r"[{}]|JSON|json|字段|系统", desc):
                flags.append("[破墙?]")
            if not (50 <= len(desc) <= 200):
                flags.append(f"[长度 {len(desc)} 字]")
        else:
            flags.append("[after 非 JSON]")
        flag_count += bool(flags)

        extra = []
        if obj:
            ev = obj.get("open_event")
            if isinstance(ev, dict) and ev.get("title"):
                extra.append(f"事件：{ev['title']}")
            sp = obj.get("spellcard_result")
            if isinstance(sp, dict) and sp.get("spellcard_name"):
                extra.append(f"符卡：{sp['spellcard_name']}（{sp.get('outcome', '?')}）")
        lines += [
            f"## #{rank}（样本 {i}）NPC：{npc} {' '.join(flags)}",
            "",
            f"**玩家输入**：{e.get('input', '') or '（空）'}",
            "",
            f"**description**：{desc}",
            "",
        ]
        if extra:
            lines += [f"**{' ｜ '.join(extra)}**", ""]

    lines += ["---", f"预筛标记 {flag_count}/{args.n} 条，需人工复核确认是否真问题。"]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"写入 {args.out}，预筛标记 {flag_count}/{args.n}")


if __name__ == "__main__":
    main()
