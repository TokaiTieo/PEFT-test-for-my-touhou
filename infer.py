"""方案 B：合并 LoRA adapter + 推理验证（《东方异变录》NPC 对话格式）。

输出必须是游戏契约 JSON（description/is_dead/relationship_update/... 字段）。
本脚本顺带做两件检查：JSON 可解析性 + 必填字段完整性。

用法：
    python infer.py                          # 用默认 adapter 路径
    python infer.py --adapter saves/xxx      # 指定 adapter
    python infer.py --base-only              # 不挂 adapter，看微调前效果（对比用）
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REQUIRED_KEYS = ["description", "is_dead"]

SAMPLE_SYSTEM = (
    "你是游戏中的NPC。请根据以下信息回应玩家。\n\n"
    "## 世界观\n幻想乡是被「博丽大结界」隔离于外界的秘境，妖怪、神明与人类共存，"
    "一切争斗必须以「符卡规则」的弹幕形式解决，而非直接杀戮。\n\n"
    "## 当前NPC信息\n姓名：博丽灵梦；身份：博丽神社巫女，幻想乡的平衡守护者；"
    "性格：懒散怕麻烦，缺钱却不庸俗，处理异变时冷静得近乎无情；当前态度：警惕但好奇。\n\n"
    "## 当前场景\n博丽神社，午后，塞钱箱前。\n\n"
    "## 对话历史\n（无）\n\n"
    "## 规则（摘要）\n1. 以博丽灵梦的身份回应，动作用（）包括起来，语言用‘’包括起来；"
    "2. description 控制在50-200字；3. 只输出JSON。"
)
SAMPLE_USER = "玩家：（往塞钱箱里投了一枚金币）‘巫女小姐，这点心意请收下。’"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2-1.5B-Instruct")
    ap.add_argument("--adapter", default="saves/qwen2-1.5b-npc-hf")
    ap.add_argument("--base-only", action="store_true", help="不加载 adapter，跑基座模型")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if not args.base_only:
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    msgs = [{"role": "system", "content": SAMPLE_SYSTEM},
            {"role": "user", "content": SAMPLE_USER}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(ids, max_new_tokens=512, temperature=0.7, top_p=0.9)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(text)

    # ---- 契约合规检查 ----
    try:
        obj = json.loads(text)
        missing = [k for k in REQUIRED_KEYS if k not in obj]
        print(f"\n[检查] JSON 可解析 ✓；缺失必填字段: {missing if missing else '无'}")
        print(f"[检查] description 长度: {len(obj.get('description', ''))} 字（规则 50-200）")
    except json.JSONDecodeError as e:
        print(f"\n[检查] JSON 解析失败 ✗：{e}")


if __name__ == "__main__":
    main()
