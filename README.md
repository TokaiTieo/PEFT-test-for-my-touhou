# Qwen2-1.5B NPC 对话垂域微调（《东方异变录》适配版）

> 目标：为同人互动游戏[《东方异变录》](https://github.com/TokaiTieo/touhou)微调一个本地 NPC 对话模型，
> 作为 DeepSeek API 之外的本地备选模型（离线兜底 / 降低成本）。
> 基座模型：Qwen2-1.5B-Instruct
> 硬件分工：本地 RTX 3060 Laptop 6GB = 跑通流程（smoke test）；算力云 RTX 4090 24GB = 正式训练；RTX 3050 4GB = 推理测试

⚠️ RTX 30 系支持 bf16；Windows 下 bitsandbytes 4bit 坑多，**本地训练不做 QLoRA**。当前两份历史命名为 `qlora` 的 YAML 实际都关闭了 4bit，运行的是普通 LoRA；只有显式启用 `quantization_bit: 4` 才是 QLoRA。

## 与通用 NPC 微调的关键差异

游戏里的 NPC 对话**不是纯文本生成**，而是严格契约：

- 输入：`prompts/npc_dialogue.txt` 模板填充后的长 prompt（世界观 / 当前 NPC 信息 / 玩家信息 / 场景 / 对话历史 / 长期剧情摘要 / 态度集合 / NPC 记忆 / 线索板 / 玩家输入 / 规则）。游戏后端把整段 prompt 作为**单条 `user` 消息**发送，不使用单独的 `system` 消息。
- 输出：**只输出 JSON**，必填 `description`、`is_dead`，可选 `relationship_update`、`memory_updates`、`task_updates`、`spellcard_result`、`exit_dialogue` 等 14 个字段
- 文风规则：动作用（）包括、语言用‘’包括、`description` 50-200 字、遵守符卡规则

因此微调目标 = **人设一致性 + 契约 JSON 合规率**双达标。

## 仓库结构

```
├── README.md
├── requirements.txt
├── train_sft.py                 # 方案 B：Transformers + PEFT 手写训练脚本
├── infer.py                     # 合并 LoRA + 推理验证（含契约合规自检）
├── configs/
│   ├── train_npc_qlora.yaml         # 方案 A：LLaMA-Factory 云端 4090 正式训练（cutoff 4096）
│   ├── train_npc_qlora_smoke.yaml   # 方案 A：本地 3060 smoke test
│   └── dataset_info.snippet.json    # LLaMA-Factory 数据集注册片段
├── data/
│   ├── npc_dialogue.json        # 训练数据（alpaca 格式，output 为契约 JSON 字符串）
│   ├── npc_eval.json            # 测试集 200 条，绝不进训练（占位）
│   └── quality_check.py         # 数据质检脚本（契约校验版）
└── eval/
    ├── eval.py                  # 微调前后对比：自动指标（JSON 合规率等）+ DeepSeek judge
    └── results/                 # 评测结果输出目录
```

## 数据格式与合成

alpaca 格式：`instruction` = 游戏真实模板填充后的 prompt（可裁剪低信息量段落），`input` = 玩家输入（也可并入 instruction），`output` = 契约 JSON 字符串。样例见 `data/npc_dialogue.json`。

- **人设来源**：游戏 `worlds/world_touhou/npcs/npc_index.json` 的真实 NPC 档案（姓名/身份/性格/口癖/符卡风格/初始态度），选 10-20 个高出场 NPC
- **世界观**：`worlds/world_touhou/worldview.txt`（可截取核心段落，保持各样本一致让模型记住）
- **场景分布**：日常对话 / 供奉与赠礼 / 任务线索推进 / 关系升降 / 符卡挑战 / 威胁冲突（触发 `exit_dialogue`）/ 值得记住的事（触发 `memory_updates`）——各字段的触发情况都要有足够样本
- 数据量目标 3000-5000 条，多轮对话占 30%+

当前 `npc_dialogue.json` 只是 4 条可执行的种子样例，用于验证格式和流程，不能代表可训练的数据规模；`npc_eval.json` 仍为空占位文件。

### 已按游戏后端确认的三个子结构

核对基准：`touhou` 仓库 `83ab94c` 的 `backend/services/ai_contracts.py`、`npc_memory_service.py`、`game_rules.py` 与 `turn_orchestrator.py`。

```json
{
  "memory_updates": [
    {
      "npc_name": "博丽灵梦",
      "summary": "玩家与灵梦共同稳定了神社附近的结界。",
      "tags": ["异变", "合作"],
      "importance": 8,
      "emotion": "信任",
      "knowledge_type": "direct",
      "source_npc": null,
      "confidence": 0.9,
      "truth_status": "accepted",
      "fact_key": null
    }
  ],
  "open_event": {
    "title": "结界波纹",
    "type": "异变线索",
    "scene": "博丽神社",
    "npc_name": "博丽灵梦",
    "description": "后山出现了可以继续追踪的异常波纹。",
    "hooks": ["前往后山调查", "询问灵梦"]
  },
  "spellcard_result": {
    "opponent": "雾雨魔理沙",
    "spellcard_name": "恋符「Master Spark」",
    "outcome": "胜利",
    "summary": "玩家抓住弹幕间隙，迫使魔理沙按规则认输。",
    "cost": "消耗少量灵力并产生轻微疲劳"
  }
}
```

- `memory_updates` 必须是**对象数组**，不是字符串数组。最小有效对象是 `npc_name` + `summary`；`tags` 等字段有后端默认值。训练样本推荐至少提供 `tags`，重要事件再给 `importance` / `emotion`。`knowledge_type` 只能是 `direct/reported/inferred/system`，`confidence` 为 0-1，`truth_status` 只能是 `accepted/disputed/superseded`。记忆落盘时 `summary` 会截到 300 字。
- `open_event` 是对象或 `null`。入口模型目前接受任意字典，但持久化只消费上例这些键，并要求 `title` 或 `description` 至少一个非空；训练统一使用完整结构，避免生成“能解析但不会落盘”的空事件。
- `spellcard_result` 是对象或 `null`。AI 侧只应生成上例 5 个键；`metrics`、`mastery`、`rule_source` 由后端确定性规则补充，不应放进训练答案。实际战斗中后端会覆盖 `opponent/spellcard_name/outcome/cost`，AI 的 `summary` 主要负责叙事补充，因此训练答案必须服从 prompt 末尾的“后端游戏规则预裁定”。

后端 JSON 清理层容忍 UTF-8 BOM、最外层 Markdown 代码块、JSON 前后的解释文字和控制字符，但**不修复尾逗号或其他非法 JSON**。训练集仍只教原始严格 JSON，不能把容错层当成输出规范。`TOUHOU_LANGGRAPH=0` 的直连回退路径也调用同一个 `parse_turn_response`，契约完全相同。

### 数据质检规则（`data/quality_check.py`）

1. output 必须是可解析 JSON，并使用固定 14 字段布局
2. `description` 长度 50-200 字（游戏规则）
3. `memory_updates` / `open_event` / `spellcard_result` 子结构与后端一致
4. 重复 output 过滤
5. 敏感词过滤
6. instruction 结构完整（含「当前NPC信息」「玩家的动作和语言」段落）

## 方案 A：LLaMA-Factory（主用）

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

1. 把 `data/npc_dialogue.json` 复制到 LLaMA-Factory 的 `data/` 下，按 `configs/dataset_info.snippet.json` 注册
2. 云端 4090：`llamafactory-cli train configs/train_npc_qlora.yaml`
3. 本地 3060 跑通流程：`llamafactory-cli train configs/train_npc_qlora_smoke.yaml`（cutoff 1024 会截断长 prompt，属预期，只验证流程）
4. 合并权重 / 聊天验证命令见 yaml 内注释

## 方案 B：Transformers + PEFT 手写版（理解原理）

```bash
pip install -r requirements.txt
python train_sft.py     # LoRA SFT，只监督 output 部分
python infer.py         # 合并 LoRA 推理 + 契约合规自检
```

手写脚本会从中间裁剪超长 prompt，保留开头的人设/世界观与结尾的玩家输入/规则，并始终完整保留监督答案。默认长度 4096，可用 `NPC_MAX_LENGTH=8192` 等环境变量调整。不要用简单的右截断，否则长 prompt 可能把整段 output 截掉，产生零监督样本。

## 接入游戏（零代码改动）

游戏 AI 层是 OpenAI 兼容客户端，走环境变量配置。本地起 OpenAI 兼容服务即可：

```bash
# 方式一：vLLM（云端 / 显存充足）
python -m vllm.entrypoints.openai.api_server --model models/qwen2-1.5b-npc-merged --port 8001

# 方式二：llama.cpp（3050 4GB，先转 GGUF 再 4bit 量化）
```

然后设置：

```bash
DEEPSEEK_BASE_URL=http://127.0.0.1:8001/v1
DEEPSEEK_MODEL=qwen2-1.5b-npc-merged
DEEPSEEK_API_KEY=local   # 本地服务任意值
```

## 评测

```bash
export DEEPSEEK_API_KEY=sk-xxx
python eval/eval.py --adapter saves/qwen2-1.5b-npc-hf
```

- **自动指标**（不调 API）：严格 JSON 可解析率 / 核心字段完整率 / 14 字段完整率 / 三个关键子结构合规率 / description 长度合规率
- **judge**：DeepSeek 对 description 三维 1-5 分（人设一致性 / 剧情连贯性 / 回复质量）
- 输出 `eval/results/compare.md` 对比表

## 执行顺序与成本

| 步骤 | 在哪跑 | 预计耗时 |
|---|---|---|
| 环境安装 + 200 条 smoke test | 本地 3060 | 半天 |
| 数据合成 3-5k 条 + 质检 | DeepSeek API | 3-5 天（成本 < 20 元） |
| 正式 SFT（3 epoch，cutoff 4096） | 算力云 4090 | 约 3-5 小时（< 20 元） |
| 评测 + 报告 | 本地 + API | 1-2 天 |
| 可选 DPO（第 4 周） | 算力云 4090 | 约 1-2 小时 |

DPO：同一套配置改三处——`stage: dpo`、`dataset: npc_preference`（chosen/rejected 格式，可用"契约合规 vs 不合规"构造偏好对）、加 `pref_beta: 0.1`。

## 训练记录

### Loss 曲线

<!-- TODO: 训练完成后贴 LLaMA-Factory plot_loss 输出 -->

### 微调前后评测对比

<!-- TODO: 贴 eval/results/compare.md -->

### 踩坑记录

<!-- TODO: 训练过程中持续补充 -->
