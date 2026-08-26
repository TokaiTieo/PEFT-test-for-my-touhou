"""方案 B：HuggingFace Transformers + PEFT 手写 LoRA SFT 训练脚本。

要点：
- 与游戏后端一致，把完整 prompt 作为单条 user 消息送入 Qwen chat template
- 只监督 output 部分（prompt 段 labels 置 -100）
- LoRA target 覆盖 attention + MLP 全部投影层，等价 LLaMA-Factory 的 lora_target: all
- 本地 3060 6GB 默认 batch 1；云端 4090 把 per_device_train_batch_size 改 4、grad accum 改 4
"""

import inspect
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2-1.5B-Instruct"   # 或本地路径
DATA = "data/npc_dialogue.json"
EVAL_DATA = "data/npc_eval.json"
OUT = "saves/qwen2-1.5b-npc-hf"
MAX_LENGTH = int(os.environ.get("NPC_MAX_LENGTH", "4096"))

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    # 云端/本地 6GB 都不用量化；真缺显存再加 load_in_4bit=True（Linux）
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# ---- LoRA ----
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # 等价 LLaMA-Factory 的 lora_target: all
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()   # 约 0.9% 参数，截图放 README

# ---- 数据：消息角色与 touhou/backend/services/ai_service.py 保持一致 ----
def user_prompt(ex):
    instruction = str(ex.get("instruction") or "").strip()
    player_input = str(ex.get("input") or "").strip()
    return instruction if not player_input else f"{instruction}\n\n{player_input}"


def to_text(ex):
    msgs = [{"role": "user", "content": user_prompt(ex)}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    answer = ex["output"] + tokenizer.eos_token
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(a_ids) >= MAX_LENGTH:
        raise ValueError(
            f"单条 output 有 {len(a_ids)} tokens，已达到 cutoff {MAX_LENGTH}；"
            "不能截断监督答案，请缩短 output 或提高 NPC_MAX_LENGTH。"
        )

    # 长 prompt 从中间裁剪，保留人设/世界观开头与玩家输入/输出规则结尾。
    # 绝不能直接对 p_ids + a_ids 右截断，否则可能没有任何 answer token 参与训练。
    prompt_budget = MAX_LENGTH - len(a_ids)
    if len(p_ids) > prompt_budget:
        head_budget = prompt_budget // 3
        p_ids = p_ids[:head_budget] + p_ids[-(prompt_budget - head_budget):]
    input_ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }

raw_ds = load_dataset("json", data_files=DATA, split="train")
ds = raw_ds.map(to_text, remove_columns=raw_ds.column_names)
raw_eval_ds = load_dataset("json", data_files=EVAL_DATA, split="train")
eval_ds = raw_eval_ds.map(to_text, remove_columns=raw_eval_ds.column_names)

training_kwargs = dict(
    output_dir=OUT,
    per_device_train_batch_size=1,        # 本地 3060；云端改 4
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,       # 云端改 4
    learning_rate=1e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    seed=42,
    data_seed=42,
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    bf16=True,
    report_to="none",
)
# Transformers 新版使用 eval_strategy，旧版使用 evaluation_strategy。
strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
training_kwargs[strategy_key] = "steps"
args = TrainingArguments(**training_kwargs)

trainer = Trainer(
    model=model, args=args, train_dataset=ds, eval_dataset=eval_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8),
)
trainer.train()
trainer.save_model(OUT)                   # 只存 LoRA adapter
