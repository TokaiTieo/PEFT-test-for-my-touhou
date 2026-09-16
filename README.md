# 《东方异变录》NPC 模型微调

这个仓库用来为我自己的同人互动游戏[《东方异变录》](https://github.com/TokaiTieo/touhou)微调 `Qwen2-1.5B-Instruct`，让它能够在游戏里充当本地 NPC 模型。

它不是一个普通的聊天微调项目。游戏要求模型一边扮演灵梦、魔理沙、咲夜等角色，一边返回后端能够直接处理的 JSON。模型说得像不像角色很重要，但 JSON 写错了，整回合就可能无法写入存档。

项目的目标很直接：

- 没网或不想调用云端 API 时，可以在本地继续游戏；
- NPC 的语气、人设和世界观尽量稳定；
- 关系、记忆、开放事件和符卡结果能够按游戏契约落盘。

## 目前做到哪了

目前正在重写训练数据，**还不能把原有 3000 条当成高质量正式训练集**。旧数据主要由模板程序拼装：即使句子不完全相同，也可能只是换名字、物品或措辞。之前仅凭条数和 JSON 合规就建议开始正式训练，不够严谨。

| 项目 | 当前状态 |
|---|---|
| 后端 JSON 契约 | 保留原有 14 字段训练格式；当前对话契约实际忽略 `is_dead`、`new_location` |
| 训练数据 | 3000 个位置中已逐条重写 1000 条，其余 2000 条保留但不进入默认训练 |
| 人设依据 | 对照 THBWiki 收录的官方资料，记录 92 个来源页面及修订号；4 条评测另有身份推断说明 |
| 数据质检 | JSON/子结构、重复对白、跨集合文本重叠、重建一致性；不等同于人工验收 |
| 手写 LoRA 训练 | 已实现，只对模型答案计算 loss |
| 训练配置 | 默认只接入 1000 条已重写子集，供试训，不混入待改旧数据 |
| 评测集 | 全部 200 条已重写并接入默认评测入口 |
| 训练后权重 | 尚未发布 |

累计重写 1000 条训练、200 条评测，分别覆盖 49、97 个项目 NPC 名称（包含别名，不等于独立角色数）。每条场景、玩家输入和完整回答直接撰写，不调用付费模型 API，不使用台词模板扩充数量。脚本只负责填充 JSON 的空字段、组装数据和校验，不负责创作对白。

有 4 条评测需要特别说明：项目中的“天独铭”与“天肱”分别使用 `npc_enoko`、`npc_biten`。据此暂按三头慧之子、孙美天修正人设，保留原名称与数据位置；这是基于项目 ID 的推断，不是官方别名，也尚未经用户确认。详见 [`data/legacy_identity_mapping.json`](data/legacy_identity_mapping.json)。未改动游戏 NPC 或存档。

`reviewed` 表示助手对照来源重写并检查过，**不是用户已验收，也不是官方对白**。具体剧情是为游戏新写的虚构情境；THBWiki 的二创梗不作为官方事实，《求闻史纪》等书中角色的推测也不直接当成真相。

逐条进度、原样本指纹和来源见 [`data/canon_review_manifest.json`](data/canon_review_manifest.json)。尚未写入该清单的索引均为待重写；[`data/dataset_manifest.json`](data/dataset_manifest.json) 仅保留旧版模板数据的历史配额，不再代表当前质量状态。

## 先跑一次数据检查

```bash
git clone https://github.com/TokaiTieo/PEFT-test-for-my-touhou.git
cd PEFT-test-for-my-touhou
python scripts/build_curated_dataset.py
python data/quality_check.py data/npc_dialogue_reviewed.json data/npc_eval_reviewed.json
```

正常情况下会看到：

```text
data/npc_dialogue_reviewed.json: 共检查 1000 条样本
description 唯一文本 1000/1000
全部通过 ✓
data/npc_eval_reviewed.json: 共检查 200 条样本
description 唯一文本 200/200
全部通过 ✓
npc_dialogue_reviewed.json ↔ npc_eval_reviewed.json: exact prompt 重叠 0
npc_dialogue_reviewed.json ↔ npc_eval_reviewed.json: exact output 重叠 0
npc_dialogue_reviewed.json ↔ npc_eval_reviewed.json: exact description 重叠 0
```

质检脚本会检查：

- output 能否被标准 JSON 解析器读取；
- 14 个顶层字段是否齐全；
- `description` 是否在 50–200 字之间；
- `memory_updates`、`open_event`、`spellcard_result` 是否符合后端结构；
- instruction 是否包含 NPC 信息和玩家输入；
- 是否存在完全重复的 output；
- 训练集和评测集之间是否有完全相同的 prompt、output 或 description。

重写子集和完整文件目前均通过结构检查，跨训练/评测的完全相同 prompt、output、description 均为 0。但完整训练文件中仍有 2000 条旧模板数据，结构通过不代表内容合格。字符相似度也只用于找疑似近重复，不能证明语义隔离或人物塑造质量。检查结果与局限见 [`data/CANON_REWRITE.md`](data/CANON_REWRITE.md)。

## 数据集是怎么来的

旧版 [`scripts/generate_dataset.py`](scripts/generate_dataset.py) 读取 NPC 档案和存档锚点拼装模板，现已停用直接执行，避免覆盖重写成果。源码留作审计，不再把它的产出叫作精写数据。

新的内容源在 [`data/curated/`](data/curated/)。每行分别是集合、NPC、案例名、场景、玩家输入、完整回答和明确的状态变化。没有替换词库，也没有循环拼接对白；固定 JSON 空字段不是台词模板。

[`data/canon_profiles.json`](data/canon_profiles.json) 及同目录 `canon_profiles_002.json` 至 `canon_profiles_016.json` 分开记录官方事实摘要、创作语气建议和需避免的二创刻板印象。页面版本与出处在审校清单中；整页 Wiki 抓取内容只放忽略提交的 `.cache/canon/`，不复制进训练答案。

编辑完成后组装并检查：

```bash
python scripts/build_curated_dataset.py --write
python scripts/build_curated_dataset.py
python data/quality_check.py data/npc_dialogue_reviewed.json data/npc_eval_reviewed.json
python -m unittest discover -s tests -v
python scripts/verify_rewrite_preservation.py
python scripts/check_backend_contract.py --touhou-root ../touhou
```

新样本只替换对应角色原有位置，其余内容保留。组装器会校验数量、原文件是否被单独修改、输出结构和近重复；默认无 `--write` 时只读检查。增加新角色前需先核查该角色官方来源，不要把未知名字或项目自设强行认定为官方角色。

旧存档锚点 [`data/saveback_anchors.json`](data/saveback_anchors.json) 仍保留供后续逐条改写参考。需要重新提取时运行：

```bash
python scripts/extract_saveback_anchors.py --save-root /path/to/touhousaveback
```

复盘旧战斗不等于新结算，重提旧约定不等于新记忆。只有本回合真正发生的变化才写入对应字段，这是重写中一并修正的原则。

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

原规模训练文件 [`data/npc_dialogue.json`](data/npc_dialogue.json) 仍是 1000 条重写与 2000 条旧数据的混合文件；[`data/npc_eval.json`](data/npc_eval.json) 的 200 条已全部重写。试训统一使用 `npc_dialogue_reviewed.json` 和 `npc_eval_reviewed.json`，不要把混合训练文件直接用于正式训练。

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

可用来试训当前已重写子集；完整数据重写和用户验收尚未完成。

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

然后：

1. 把本仓库的 `data/npc_dialogue_reviewed.json` 和 `data/npc_eval_reviewed.json` 复制到 LLaMA-Factory 的 `data/`；
2. 把 `configs/dataset_info.snippet.json` 中的条目合并进 LLaMA-Factory 的 `data/dataset_info.json`；
3. 使用本仓库中的 YAML 启动训练。

子集试训：

```bash
llamafactory-cli train /path/to/PEFT-test-for-my-touhou/configs/train_npc_qlora.yaml
```

本地 smoke test：

```bash
llamafactory-cli train /path/to/PEFT-test-for-my-touhou/configs/train_npc_qlora_smoke.yaml
```

两份 YAML 的名字保留了早期的 `qlora` 命名，但目前没有打开 4bit 量化，实际运行的是普通 LoRA。只有取消 `quantization_bit: 4` 的注释或显式加入该配置后，才是 QLoRA。

试训配置使用 4096 tokens；smoke 配置使用 1024 tokens。后者可能截断样本，只用于环境检查，不能用于质量结论。两份配置现在都不读取未重写数据。

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

默认使用不进入训练的 200 条 `data/npc_eval_reviewed.json`。题目同时包含应该更新和不应该更新状态的情况。多数角色只有 2 条，且两个集合由同一助手编写，不能据此获得可靠的逐角色质量结论或视为独立盲测。训练完成后运行：

```bash
python eval/eval.py --adapter saves/qwen2-1.5b-npc-hf --skip-judge
```

自动评测包括：

- JSON 可解析率；
- 核心字段完整率；
- 14 字段完整率；
- 三个关键子结构合规率；
- description 长度合规率。

默认不调用付费 API，即使环境中有 Key 也不会自动调用。旧的付费 judge 功能只有主动传入 `--paid-judge` 才启用，本轮重写及校验没有使用它。自动格式指标不能替代人设审阅或游戏内实测。

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
│   ├── npc_dialogue.json       # 3000 条，新旧混合，迁移中
│   ├── npc_eval.json           # 200 条，已全部重写
│   ├── npc_dialogue_reviewed.json # 1000 条直接重写的试训子集
│   ├── npc_eval_reviewed.json  # 200 条直接重写的评测集
│   ├── curated/               # 逐条撰写原稿，不由模板生成
│   ├── canon_profiles.json    # 官方事实摘要与创作约束
│   ├── canon_review_manifest.json # 来源、进度、逐条指纹和诊断
│   ├── saveback_anchors.json   # 测试存档的匿名化剧情锚点
│   ├── dataset_manifest.json   # 旧模板数据的历史配额
│   └── quality_check.py        # 数据契约质检
├── eval/
│   ├── eval.py                 # 微调前后对比
│   └── results/                # 评测输出
├── infer.py                    # 单条推理与 JSON 检查
├── scripts/
│   ├── extract_saveback_anchors.py # 从测试存档提取匿名化锚点
│   ├── build_curated_dataset.py # 只组装/验证直接撰写内容
│   └── generate_dataset.py     # 已停用入口，仅保留审计记录
├── train_sft.py                # Transformers + PEFT 训练脚本
└── requirements.txt
```

## 接下来要做的事

- 本轮要求的 1000 条训练和全部 200 条评测已重写；剩余 2000 条旧训练数据不在本轮范围，也不进入默认训练；
- 请用户确认“天独铭 / 天肱”的项目 ID 身份映射；
- 请用户审阅这批对白，检查是否仍有过于相似的节奏、说教口气或缺乏人物特征的回答；
- 增补多轮冲突、正反状态变化、交易等覆盖；再做实际基座/微调对比；
- 增加近似语义去重和按角色留出的泛化评测，而不只检查完全重复；
- 统计真实 prompt 的 token 分布，再决定正式训练使用 4096 还是更长上下文；
- 完成基座与微调模型对比后，再决定是否需要 DPO。

