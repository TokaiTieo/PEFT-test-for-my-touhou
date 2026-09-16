"""统计 generations.json 中去 markdown 围栏后的 JSON 可解析率。

用途：区分基座模型的失败模式——"不会输出 JSON" vs "输出了但包了 ```json 围栏"。
用法：python3 eval/fence_check.py [generations.json 路径]
"""
import json
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "eval/results/generations.json"
data = json.load(open(path, encoding="utf-8"))

for key in ("before", "after"):
    cnt = 0
    for x in data:
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", x[key].strip())
        try:
            json.loads(t)
            cnt += 1
        except Exception:
            pass
    print(key, f"{cnt / len(data):.1%}")
