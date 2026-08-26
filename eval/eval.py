"""评测脚本：微调前 vs 微调后（《东方异变录》契约版）。

流程：
1. data/npc_eval.json 200 条，基座模型和微调后模型各生成一遍
2. 自动指标（不调 API）：
   - JSON 可解析率
   - 核心字段完整率（description / is_dead）
   - 14 字段完整率与三个关键子结构合规率
   - description 长度合规率（50-200 字）
3. DeepSeek API 做 judge，三维 1-5 分：人设一致性 / 剧情连贯性 / 回复质量
   （judge 对象为 description 字段）
4. 输出对比表 → eval/results/compare.md（README 主图素材）

用法：
    export DEEPSEEK_API_KEY=sk-xxx
    python eval/eval.py --adapter saves/qwen2-1.5b-npc-hf
    python eval/eval.py --adapter saves/xxx --skip-judge   # 只跑自动指标
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REQUIRED_KEYS = ["description", "is_dead"]
FULL_KEYS = [
    "description", "is_dead", "time_cost", "new_energy_state", "new_location",
    "relationship_update", "inventory_updates", "reputation_updates", "world_effects",
    "task_updates", "memory_updates", "open_event", "spellcard_result", "exit_dialogue",
]
DESC_MIN, DESC_MAX = 50, 200
DIMS = ["人设一致性", "剧情连贯性", "回复质量"]

JUDGE_PROMPT = """你是游戏 NPC 对话质量评审。请根据以下信息给 NPC 回复打分。

【NPC 设定与场景】
{instruction}

【玩家输入】
{player_input}

【NPC 回复】
{description}

请从三个维度各打 1-5 分（5 最高），只输出 JSON：
{{"人设一致性": x, "剧情连贯性": x, "回复质量": x}}"""


def generate(model, tok, samples, max_new_tokens=512):
    outs = []
    for s in samples:
        prompt = str(s["instruction"]).strip()
        if str(s.get("input") or "").strip():
            prompt += "\n\n" + str(s["input"]).strip()
        # 与游戏 backend 的 OpenAI 调用一致：完整 prompt 是单条 user 消息。
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        outs.append(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    return outs


def auto_metrics(outputs):
    n = len(outputs)
    parse_ok = field_ok = full_ok = nested_ok = len_ok = 0
    for t in outputs:
        try:
            obj = json.loads(t)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        parse_ok += 1
        if all(k in obj for k in REQUIRED_KEYS):
            field_ok += 1
            if DESC_MIN <= len(obj.get("description", "")) <= DESC_MAX:
                len_ok += 1
        if all(k in obj for k in FULL_KEYS):
            full_ok += 1
        memories = obj.get("memory_updates")
        event = obj.get("open_event")
        battle = obj.get("spellcard_result")
        memories_ok = isinstance(memories, list) and all(
            isinstance(item, dict) and item.get("npc_name") and item.get("summary")
            for item in memories
        )
        event_ok = event is None or (
            isinstance(event, dict)
            and all(event.get(key) for key in ("title", "type", "scene", "description"))
            and isinstance(event.get("hooks"), list)
            and all(isinstance(hook, str) for hook in event["hooks"])
        )
        battle_ok = battle is None or (
            isinstance(battle, dict)
            and all(battle.get(key) for key in ("opponent", "spellcard_name", "outcome", "summary"))
            and "cost" in battle
        )
        if memories_ok and event_ok and battle_ok:
            nested_ok += 1
    return {
        "JSON可解析率": parse_ok / n,
        "核心字段完整率": field_ok / n,
        "14字段完整率": full_ok / n,
        "关键子结构合规率": nested_ok / n,
        "长度合规率": len_ok / n,
    }


def judge(samples, outputs):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    scores = {d: [] for d in DIMS}
    for s, t in zip(samples, outputs):
        try:
            desc = json.loads(t).get("description", t)
        except json.JSONDecodeError:
            desc = t  # 解析失败也送审，分数自然会低
        resp = client.chat.completions.create(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                instruction=s["instruction"], player_input=s.get("input", ""), description=desc)}],
            temperature=0,
        )
        m = re.search(r"\{.*\}", resp.choices[0].message.content, re.S)
        try:
            obj = json.loads(m.group(0))
            for d in DIMS:
                scores[d].append(float(obj[d]))
        except (AttributeError, json.JSONDecodeError, KeyError):
            continue  # 单条 judge 失败跳过
    return {d: (sum(v) / len(v) if v else 0.0) for d, v in scores.items()}


def fmt_table(rows):
    lines = ["| 指标 | 微调前 | 微调后 |", "|---|---|---|"]
    for name, before, after in rows:
        b = f"{before:.2%}" if isinstance(before, float) and before <= 1 else f"{before:.2f}"
        a = f"{after:.2%}" if isinstance(after, float) and after <= 1 else f"{after:.2f}"
        lines.append(f"| {name} | {b} | {a} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2-1.5B-Instruct")
    ap.add_argument("--adapter", default="saves/qwen2-1.5b-npc-hf")
    ap.add_argument("--data", default="data/npc_eval.json")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    samples = json.load(open(args.data, encoding="utf-8"))
    assert samples, "npc_eval.json 为空，先合成 200 条测试集（绝不进训练）"
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    # 微调前
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    out_before = generate(base, tok, samples)
    del base
    torch.cuda.empty_cache()

    # 微调后
    ft = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    ft = PeftModel.from_pretrained(ft, args.adapter)
    ft = ft.merge_and_unload()
    out_after = generate(ft, tok, samples)

    m_before, m_after = auto_metrics(out_before), auto_metrics(out_after)
    rows = [(k, m_before[k], m_after[k]) for k in m_before]

    if not args.skip_judge:
        j_before, j_after = judge(samples, out_before), judge(samples, out_after)
        rows += [(f"judge-{d}", j_before[d], j_after[d]) for d in DIMS]

    table = fmt_table(rows)
    print("\n" + table)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "compare.md", "w", encoding="utf-8") as f:
        f.write("# 微调前后对比\n\n" + table + "\n")
    with open(Path(args.out) / "generations.json", "w", encoding="utf-8") as f:
        json.dump([{"instruction": s["instruction"], "input": s.get("input", ""),
                    "before": b, "after": a}
                   for s, b, a in zip(samples, out_before, out_after)],
                  f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {args.out}/compare.md 与 generations.json")


if __name__ == "__main__":
    main()
