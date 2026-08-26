"""方案 B：HuggingFace Transformers + PEFT 手写 LoRA SFT 训练脚本。

要点：
- 用 Qwen 官方 chat template 拼文本，只监督 output 部分（prompt 段 labels 置 -100）
- LoRA target 覆盖 attention + MLP 全部投影层，等价 LLaMA-Factory 的 lora_target: all
- 本地 3060 6GB 默认 batch 1；云端 4090 把 per_device_train_batch_size 改 4、grad accum 改 4
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2-1.5B-Instruct"   # 或本地路径
DATA = "data/npc_dialogue.json"
OUT = "saves/qwen2-1.5b-npc-hf"

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

# ---- 数据：用 Qwen 官方 chat template 拼文本，只监督 output 部分 ----
def to_text(ex):
    msgs = [{"role": "system", "content": ex["instruction"]},
            {"role": "user", "content": ex["input"]}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    answer = ex["output"] + tokenizer.eos_token
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    input_ids = (p_ids + a_ids)[:4096]   # 游戏真实 prompt 很长，云端 4096；本地 smoke 改 1024
    labels = ([-100] * len(p_ids) + a_ids)[:4096]   # prompt 部分不算 loss
    return {"input_ids": input_ids, "labels": labels}

ds = load_dataset("json", data_files=DATA, split="train").map(
    to_text, remove_columns=["instruction", "input", "output"])

args = TrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=1,        # 本地 3060；云端改 4
    gradient_accumulation_steps=16,       # 云端改 4
    learning_rate=1e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=200,
    bf16=True,
    report_to="none",
)

trainer = Trainer(
    model=model, args=args, train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
)
trainer.train()
trainer.save_model(OUT)                   # 只存 LoRA adapter
