# 《东方异变录》NPC 模型微调

这个仓库用来为我自己的同人互动游戏[《东方异变录》](https://github.com/TokaiTieo/touhou)微调 `Qwen2-1.5B-Instruct`，让它能够在游戏里充当本地 NPC 模型。

它不是一个普通的聊天微调项目。游戏要求模型一边扮演灵梦、魔理沙、咲夜等角色，一边返回后端能够直接处理的 JSON。模型说得像不像角色很重要，但 JSON 写错了，整回合就可能无法写入存档。

项目的目标很直接：

- 没网或不想调用云端 API 时，可以在本地继续游戏；
- NPC 的语气、人设和世界观尽量稳定；
- 关系、记忆、开放事件和符卡结果能够按游戏契约落盘。

## 训练结果

第一轮基线已完成：AutoDL RTX 3090 24G，LLaMA-Factory，LoRA r16/alpha32，bf16，cutoff 4096，等效 batch 16，3 epochs（564 步）。1.5B 训练 43 分钟，7B 约 2 小时。

### 主结果：Qwen2-1.5B-Instruct + LoRA（checkpoint-564）

| 指标 | 微调前 | 微调后 |
|---|---|---|
| JSON 可解析率 | 4.50% | **98.00%** |
| 核心字段完整率 | 1.50% | **98.00%** |
| 14 字段完整率 | 0.50% | **98.00%** |
| 关键子结构合规率 | 0.00% | **98.00%** |
| 长度合规率 | 0.00% | **97.50%** |

基座的主要失败模式是给 JSON 包 Markdown 围栏：去掉围栏后基座可解析率为 85.5%，但游戏后端要求输出可被 `json.loads` 直接解析，因此裸解析率才是交付指标。围栏剥离统计见 [`eval/fence_check.py`](eval/fence_check.py)。

### epoch 消融（200 条独立评测集，微调后）

| checkpoint | 可解析 | 核心字段 | 14 字段 | 子结构 | 长度 |
|---|---|---|---|---|---|
| 200（ep1） | 97.0% | 97.0% | 97.0% | 97.0% | 97.0% |
| 400（ep2） | 97.5% | 97.0% | 97.0% | 97.0% | 96.5% |
| 564（ep3） | **98.0%** | **98.0%** | **98.0%** | **98.0%** | **97.5%** |

单调微升、无过拟合迹象；增益集中在 ep1，ep2/ep3 边际约 1pp。最终采用 checkpoint-564。

### 7B 对照实验（Qwen2.5-7B-Instruct + LoRA，同数据同超参）

| 指标 | 微调前 | 微调后 |
|---|---|---|
| JSON 可解析率 | 99.50% | **99.00%** |
| 核心字段完整率 | 99.50% | **99.00%** |
| 14 字段完整率 | 99.50% | **99.00%** |
| 关键子结构合规率 | 0.00% | **99.00%** |
| 长度合规率 | 43.00% | **99.00%** |

7B 基座的失败模式与 1.5B 完全不同：不包围栏，但嵌套子结构全错、长度失控——深层 schema 约束靠 prompt 给不了，只能靠 SFT。微调后 7B 仅领先 1.5B 约 1pp，考虑本地部署体积（15G vs 3G）与推理速度，**选型 1.5B**。

### 产物

| 产物 | 位置 |
|---|---|
| 1.5B LoRA 权重（选用） | [huggingface.co/TokaiTieo/qwen2-1.5b-npc-lora](https://huggingface.co/TokaiTieo/qwen2-1.5b-npc-lora) |
| 7B LoRA 权重（对照） | [huggingface.co/TokaiTieo/qwen2.5-7b-npc-lora](https://huggingface.co/TokaiTieo/qwen2.5-7b-npc-lora) |
| 1.5B 评测明细 + loss 曲线 | 分支 [`q2-1.5b`](https://github.com/TokaiTieo/PEFT-test-for-my-touhou/tree/q2-1.5b) 的 `eval/results/` |
| 7B 评测明细 + loss 曲线 | 分支 [`q2.5-7b`](https://github.com/TokaiTieo/PEFT-test-for-my-touhou/tree/q2.5-7b) 的 `eval/results/` |

## 目前做到哪了

仓库现在已经具备第一轮基线训练所需的数据和评测流程，但还没有可直接发布的成品模型。

| 项目 | 当前状态 |
|---|---|
| 后端 JSON 契约 | 已对照 `touhou` 后端代码核实 |
| 训练数据 | 3000 条：基础分层 2000 条 + 测试存档驱动 1000 条 |
| 数据质检 | 可检查 14 个顶层字段、状态更新子结构和跨集合泄漏 |
| 手写 LoRA 训练 | 已实现，只对模型答案计算 loss |
| LLaMA-Factory 配置 | 已提供正式训练和 smoke test 配置 |
| 独立评测集 | 200 条，使用与训练集互斥的玩家动作模板 |
| 训练后权重 | 已发布：[1.5B](https://huggingface.co/TokaiTieo/qwen2-1.5b-npc-lora)（选用）与 [7B](https://huggingface.co/TokaiTieo/qwen2.5-7b-npc-lora)（对照） |

也就是说：现在可以开始第一轮基线训练，并用固定评测集比较微调前后的格式遵循能力。其中 1000 条新增数据参考了真实测试存档中的连续对话、长期记忆、开放事件、符卡与关系状态；这些内容经过匿名化和去重，但仍不等同于 3000 条人工精写对白。

## 先跑一次数据检查

```bash
git clone https://github.com/TokaiTieo/PEFT-test-for-my-touhou.git
cd PEFT-test-for-my-touhou
python data/quality_check.py data/npc_dialogue.json data/npc_eval.json
```

正常情况下会看到：

```text
data/npc_dialogue.json: 共检查 3000 条样本
description 唯一文本 3000/3000
全部通过 ✓
data/npc_eval.json: 共检查 200 条样本
description 唯一文本 200/200
全部通过 ✓
npc_dialogue.json ↔ npc_eval.json: exact prompt 重叠 0
npc_dialogue.json ↔ npc_eval.json: exact output 重叠 0
```

质检脚本会检查：

- output 能否被标准 JSON 解析器读取；
- 14 个顶层字段是否齐全；
- `description` 是否在 50–200 字之间；
- `memory_updates`、`open_event`、`spellcard_result` 是否符合后端结构；
- instruction 是否包含 NPC 信息和玩家输入；
- 是否存在完全重复的 output；
- 训练集和评测集之间是否有完全相同的 prompt 或 output。

## 数据集是怎么来的

[`scripts/generate_dataset.py`](scripts/generate_dataset.py) 会读取同级 `touhou` 仓库中的真实 NPC 档案、地点表，以及已经匿名化的 [`data/saveback_anchors.json`](data/saveback_anchors.json)，用固定种子生成可复现的数据。它不联网，也不调用付费模型 API。

基础训练集按以下场景分层：日常 320、赠礼 180、协助 160、约定 160、秘密 120、开放事件 180、符卡 300、任务 180、交易 140、威胁退出 160、传闻 100，共 2000 条。

另外 1000 条来自 `touhousaveback` 的测试结果：真实对白续接 250、长期记忆召回 200、开放事件续查 150、符卡复盘 100、关系连续性 150、多状态混合 150。派生锚点包括 43 段连续对话、97 条长期记忆、14 个开放事件、5 次符卡记录和 13 个关系状态，所有玩家姓名、角色 UUID、高权限 prompt 与时间戳均已移除。评测集仍保持独立的 200 条，不吸收这些存档锚点。完整配额见 [`data/dataset_manifest.json`](data/dataset_manifest.json)。

如果 `PEFT-test-for-my-touhou` 与 `touhou` 是同级目录，直接运行：

```bash
python scripts/generate_dataset.py
python data/quality_check.py data/npc_dialogue.json data/npc_eval.json
```

如果需要从本地测试存档重新提取锚点，先运行：

```bash
python scripts/extract_saveback_anchors.py --save-root /path/to/touhousaveback
```

否则显式指定游戏仓库：

```bash
python scripts/generate_dataset.py --touhou-root /path/to/touhou
```

固定随机种子为 `20260826`。同一份源档案、锚点和生成器会得到相同数据，适合审查、复现和后续迭代。

## 一条训练数据长什么样

数据采用 Alpaca 三字段格式：

```json
{
  "instruction": "填充后的游戏 NPC prompt",
  "input": "",
  "output": "{\"description\": \"……\", \"is_dead\": false, ...}"
}
```

这里有两个容易踩坑的地方：

1. 游戏运行时会把完整 prompt 作为一条 `user` 消息发送，不会另外拆出 `system` 消息。训练和推理脚本也必须保持这个格式。
2. `output` 本身是一个 JSON 字符串，所以文件中会看到转义后的双引号。解析外层 JSON 后，output 还要能再次解析成对象。

完整训练集在 [`data/npc_dialogue.json`](data/npc_dialogue.json)，独立评测集在 [`data/npc_eval.json`](data/npc_eval.json)。

## 游戏真正需要的三个子结构

下面的结构来自 `touhou` 后端实际解析与持久化代码。

### NPC 记忆

`memory_updates` 必须是对象数组，不能写成字符串数组。

```json
{
  "memory_updates": [
    {
      "npc_name": "博丽灵梦",
      "summary": "玩家与灵梦共同稳定了神社附近的结界。",
      "tags": ["异变", "合作"],
      "importance": 8,
      "emotion": "信任"
    }
  ]
}
```

最少需要 `npc_name` 和 `summary`。建议同时提供 `tags`；重要事件再填写 `importance` 和 `emotion`。后端最终只保留 summary 的前 300 字。

高级字段包括：

- `knowledge_type`：`direct`、`reported`、`inferred` 或 `system`；
- `confidence`：0–1；
- `truth_status`：`accepted`、`disputed` 或 `superseded`；
- `source_npc`、`fact_key`：需要处理消息来源或事实冲突时再使用。

### 开放事件

没有触发事件时返回 `null`；触发时使用固定对象结构：

```json
{
  "open_event": {
    "title": "结界波纹",
    "type": "异变线索",
    "scene": "博丽神社",
    "npc_name": "博丽灵梦",
    "description": "后山出现了可以继续追踪的异常波纹。",
    "hooks": ["前往后山调查", "询问灵梦"]
  }
}
```

后端至少需要 `title` 或 `description` 才会记录事件。训练数据统一给出完整结构，避免模型生成“看起来像事件、实际不会落盘”的空对象。

### 符卡结果

没有发生战斗时返回 `null`。发生战斗时，模型只需要生成下面五个字段：

```json
{
  "spellcard_result": {
    "opponent": "雾雨魔理沙",
    "spellcard_name": "恋符「Master Spark」",
    "outcome": "胜利",
    "summary": "玩家抓住弹幕间隙，迫使魔理沙按规则认输。",
    "cost": "消耗少量灵力并产生轻微疲劳"
  }
}
```

游戏会用确定性规则重新计算对手、胜负、消耗、命中指标和熟练度。模型最重要的工作是写好 `summary`，并服从 prompt 末尾的“后端游戏规则预裁定”。不要在训练答案里生成 `metrics`、`mastery` 或 `rule_source`。

## 训练方式一：LLaMA-Factory

这是正式训练推荐使用的方式。

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

然后：

1. 把本仓库的 `data/npc_dialogue.json` 和 `data/npc_eval.json` 复制到 LLaMA-Factory 的 `data/`；
2. 把 `configs/dataset_info.snippet.json` 中的条目合并进 LLaMA-Factory 的 `data/dataset_info.json`；
3. 使用本仓库中的 YAML 启动训练。

正式训练：

```bash
llamafactory-cli train /path/to/PEFT-test-for-my-touhou/configs/train_npc_qlora.yaml
```

本地 smoke test：

```bash
llamafactory-cli train /path/to/PEFT-test-for-my-touhou/configs/train_npc_qlora_smoke.yaml
```

两份 YAML 的名字保留了早期的 `qlora` 命名，但目前没有打开 4bit 量化，实际运行的是普通 LoRA。只有取消 `quantization_bit: 4` 的注释或显式加入该配置后，才是 QLoRA。

正式配置使用 4096 tokens；smoke 配置使用 1024 tokens，只用来确认环境、数据和训练流程能否正常工作。

## 训练方式二：Transformers + PEFT

`train_sft.py` 是一份更容易阅读和修改的手写实现：

```bash
pip install -r requirements.txt
python train_sft.py
```

默认设置：

- 基座：`Qwen/Qwen2-1.5B-Instruct`；
- LoRA rank：16；
- 最大长度：4096；
- batch size：1；
- 只对 output 部分计算 loss。

可以通过环境变量调整最大长度：

```bash
NPC_MAX_LENGTH=8192 python train_sft.py
```

Windows PowerShell：

```powershell
$env:NPC_MAX_LENGTH = "8192"
python train_sft.py
```

当 prompt 太长时，脚本会保留开头的人设与世界观、结尾的玩家输入与输出规则，并从中间裁剪。模型答案始终完整保留，避免右截断把监督内容一起删掉。

## 推理检查

```bash
python infer.py --adapter saves/qwen2-1.5b-npc-hf
```

只测试基座模型：

```bash
python infer.py --base-only
```

脚本会区分两种情况：

- 严格 JSON：模型直接输出合法对象；
- 后端可恢复 JSON：例如外面包了 Markdown 代码块，游戏清理后仍能解析。

后端可以处理 BOM、最外层代码块、JSON 前后的少量说明和控制字符，但不会修复尾逗号等非法语法。训练数据只接受严格 JSON。

## 评测

仓库已经提供完全不进入训练集的 200 条 `data/npc_eval.json`。训练完成后运行：

```bash
python eval/eval.py --adapter saves/qwen2-1.5b-npc-hf --skip-judge
```

自动评测包括：

- JSON 可解析率；
- 核心字段完整率；
- 14 字段完整率；
- 三个关键子结构合规率；
- description 长度合规率。

如果设置了 DeepSeek API Key，还可以增加人设、连贯性和回复质量评分：

```bash
export DEEPSEEK_API_KEY=sk-xxx
python eval/eval.py --adapter saves/qwen2-1.5b-npc-hf
```

PowerShell 使用：

```powershell
$env:DEEPSEEK_API_KEY = "sk-xxx"
python eval/eval.py --adapter saves/qwen2-1.5b-npc-hf
```

结果会写入 `eval/results/compare.md` 和 `eval/results/generations.json`。

## 接入《东方异变录》

训练完成后，先把模型部署成 OpenAI 兼容服务，例如 vLLM 或 llama.cpp。然后修改游戏使用的环境变量：

```text
DEEPSEEK_BASE_URL=http://127.0.0.1:8001/v1
DEEPSEEK_MODEL=qwen2-1.5b-npc-merged
DEEPSEEK_API_KEY=local
```

游戏的 LangGraph 工作流和 `TOUHOU_LANGGRAPH=0` 直连回退都会经过同一个 JSON 契约解析器，因此不需要为两条路径分别训练模型。

## 仓库结构

```text
├── configs/                    # LLaMA-Factory 训练配置
├── data/
│   ├── npc_dialogue.json       # 3000 条正式训练集
│   ├── npc_eval.json           # 200 条独立评测集
│   ├── saveback_anchors.json   # 测试存档的匿名化剧情锚点
│   ├── dataset_manifest.json   # 场景配额、覆盖数和隔离结果
│   └── quality_check.py        # 数据契约质检
├── eval/
│   ├── eval.py                 # 微调前后对比
│   └── results/                # 评测输出
├── infer.py                    # 单条推理与 JSON 检查
├── scripts/
│   ├── extract_saveback_anchors.py # 从测试存档提取匿名化锚点
│   └── generate_dataset.py     # 从真实游戏档案和锚点重建数据集
├── train_sft.py                # Transformers + PEFT 训练脚本
└── requirements.txt
```

## 接下来要做的事

- 跑完第一轮基座/微调对比，记录 JSON 合规率和各类场景失分；
- 为口吻趋同或失分明显的角色补充人工精写样本；
- 增加近似语义去重和按角色留出的泛化评测，而不只检查完全重复；
- 统计真实 prompt 的 token 分布，再决定正式训练使用 4096 还是更长上下文；
- 完成基座与微调模型对比后，再决定是否需要 DPO。

