"""judge_only.py：对已有的 generations.json 做 DeepSeek judge，不重跑生成。

用途：eval.py 的 judge 与生成耦合，生成结果已归档后想补 judge 时用本脚本。
需要环境变量 DEEPSEEK_API_KEY，可选 DEEPSEEK_BASE_URL / DEEPSEEK_MODEL。

用法：
    python eval/judge_only.py eval/results/1.5b-ck564/generations.json
输出：终端表格 + 同目录 judge.md
"""
import argparse
import json
import os
import re
from pathlib import Path

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


def extract_description(text):
    """与 eval.py 口径一致：取 description 字段；解析失败送原文。多做一步去围栏。"""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(t).get("description", text)
    except json.JSONDecodeError:
        return text


def judge(entries, key, client, model):
    scores = {d: [] for d in DIMS}
    failed = 0
    for i, e in enumerate(entries):
        desc = extract_description(e[key])
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    instruction=e["instruction"], player_input=e.get("input", ""), description=desc)}],
                temperature=0,
            )
            m = re.search(r"\{.*\}", resp.choices[0].message.content, re.S)
            obj = json.loads(m.group(0))
            for d in DIMS:
                scores[d].append(float(obj[d]))
        except Exception:
            failed += 1
        if (i + 1) % 20 == 0:
            print(f"  {key}: {i + 1}/{len(entries)}", flush=True)
    avg = {d: (sum(v) / len(v) if v else 0.0) for d, v in scores.items()}
    return avg, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generations", help="generations.json 路径")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    entries = json.load(open(args.generations, encoding="utf-8"))
    results = {}
    for key in ("before", "after"):
        avg, failed = judge(entries, key, client, model)
        results[key] = (avg, failed)
        print(f"{key} 完成，失败 {failed}/{len(entries)}")

    lines = ["| 维度 | 微调前 | 微调后 |", "|---|---|---|"]
    for d in DIMS:
        lines.append(f"| {d} | {results['before'][0][d]:.2f} | {results['after'][0][d]:.2f} |")
    table = "\n".join(lines)
    print("\n" + table)

    out = Path(args.generations).parent / "judge.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# DeepSeek judge（{model}，1-5 分）\n\n{table}\n\n"
                f"失败条数：before {results['before'][1]}，after {results['after'][1]}\n")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
