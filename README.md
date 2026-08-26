# Qwen2-1.5B NPC 对话垂域微调（《东方异变录》适配版）

> 目标：为同人互动游戏[《东方异变录》](https://github.com/TokaiTieo/touhou)微调一个本地 NPC 对话模型，
> 作为 DeepSeek API 之外的本地备选模型（离线兜底 / 降低成本）。
> 基座模型：Qwen2-1.5B-Instruct
> 硬件分工：本地 RTX 3060 Laptop 6GB = 跑通流程（smoke test）；算力云 RTX 4090 24GB = 正式训练；RTX 3050 4GB = 推理测试

⚠️ RTX 30 系支持 bf16；Windows 下 bitsandbytes 4bit 坑多，**本地训练不做 QLoRA**，QLoRA 到云端 Linux 再说。

## 与通用 NPC 微调的关键差异

游戏里的 NPC 对话**不是纯文本生成**，而是严格契约：

- 输入：`prompts/npc_dialogue.txt` 模板填充后的长 prompt（世界观 / 当前 NPC 信息 / 玩家信息 / 场景 / 对话历史 / 长期剧情摘要 / 态度集合 / NPC 记忆 / 线索板 / 玩家输入 / 规则）
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

### 数据质检规则（`data/quality_check.py`）

1. output 必须是可解析 JSON，含必填字段 `description` / `is_dead`
2. `description` 长度 50-200 字（游戏规则）
3. 重复 output 过滤
4. 敏感词过滤
5. instruction 结构完整（含「当前NPC信息」「玩家的动作和语言」段落）

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

⚠️ 待验证：游戏后端对契约 JSON 的解析容错（```json 代码块包裹、尾逗号等）以 `backend/` 实际解析逻辑为准；`spellcard_result` / `open_event` / `memory_updates` 的子结构同样以实际为准，合成训练数据前先核对。

## 评测

```bash
export DEEPSEEK_API_KEY=sk-xxx
python eval/eval.py --adapter saves/qwen2-1.5b-npc-hf
```

- **自动指标**（不调 API）：JSON 可解析率 / 必填字段完整率 / description 长度合规率——这是本项目最核心的对比
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
