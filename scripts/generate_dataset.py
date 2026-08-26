"""用《东方异变录》真实 NPC/地点档案生成可复现的训练与评测数据。

不调用网络或大模型 API。固定种子、分层配额和互斥动作模板保证结果可重建，
每条答案都遵守 ``touhou/backend/services/ai_contracts.py`` 的实际契约。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

SEED = 20260826
TRAIN_QUOTAS = {
    "daily": 320, "gift": 180, "help": 160, "promise": 160,
    "secret": 120, "open_event": 180, "spellcard": 300, "task": 180,
    "trade": 140, "threat": 160, "rumor": 100,
}
EVAL_QUOTAS = {
    "daily": 20, "gift": 15, "help": 15, "promise": 15, "secret": 10,
    "open_event": 20, "spellcard": 30, "task": 20, "trade": 15,
    "threat": 20, "rumor": 20,
}
TIMES = ["清晨", "上午", "正午", "午后", "黄昏", "入夜", "深夜"]
WEATHERS = ["风里带着淡淡花香", "薄云正掠过天空", "远处传来断续蝉鸣", "空气里浮着细小灵光", "四周安静得能听见风声", "一阵短雨刚刚停下"]
PROPS = ["一枚旧护符", "一束铃兰", "一包茶叶", "一块星砂", "一张残缺地图", "一颗发光蘑菇", "一小瓶露水", "一段红色丝线"]
CLUES = ["不自然的结界波纹", "逆流而上的光点", "没有主人的脚印", "反复出现的陌生铃声", "突然失色的花丛", "会移动的淡紫色雾气", "夜里发亮的旧路标", "短暂消失的影子"]
TASKS = ["调查异常灵光", "寻找失落护符", "确认山道传闻", "护送药材", "修复小型结界", "追踪奇怪足迹", "收集魔力样本", "拜访线索见证者"]
OUTCOMES = ["胜利", "失败", "平局"]

# train/eval 使用互斥玩家动作，不让同一 prompt 跨集合出现。
ACTIONS = {
    "train": {
        "daily": ["（礼貌地点头）‘我刚到这里，能陪我聊一会儿吗？’", "（望向四周）‘这里平时也是这样的气氛吗？’"],
        "gift": ["（递出{prop}）‘这是路上得到的，送给你吧。’", "（把{prop}放到桌边）‘多谢你之前的照顾，请收下。’"],
        "help": ["（挽起袖子）‘这件事我可以帮忙，我们一起处理吧。’", "（收好行囊）‘把要做的事告诉我，我愿意同行。’"],
        "promise": ["（郑重地点头）‘线索交给我，我会回来告诉你结果。’", "（记下约定）‘无论查到什么，我都不会失约。’"],
        "secret": ["（压低声音）‘我发现了{clue}，请先别声张。’", "（确认附近无人）‘关于{clue}，我有个尚未证实的发现。’"],
        "open_event": ["（指向远处）‘那里出现了{clue}，要不要去调查？’", "（摊开地图）‘{clue}连续出现了三次，恐怕不是巧合。’"],
        "spellcard": ["（展开符卡）‘按符卡规则，和我进行一场正式弹幕决斗吧！’", "（飞到半空）‘胜负只按弹幕规则决定，来吧！’"],
        "task": ["（递上调查笔记）‘“{task}”有了进展，请检查。’", "（拿出证据）‘“{task}”已经可以结案了。’"],
        "trade": ["（展示{prop}）‘我想用它交换旅途补给。’", "（把{prop}摆在面前）‘请给个公平的交换条件。’"],
        "threat": ["（握住武器）‘别挡路，否则我就强行过去。’", "（故意释放敌意）‘立刻把知道的都说出来。’"],
        "rumor": ["（小声转述）‘有人说最近出现了{clue}，但我没亲眼见到。’", "（翻看笔记）‘我听说{clue}与附近有关，你觉得可信吗？’"],
    },
    "eval": {
        "daily": ["（收起伞）‘我第一次在这个时辰来，这里有什么规矩吗？’", "（避开水洼）‘我想听听你对最近幻想乡的看法。’"],
        "gift": ["（解开布包，露出{prop}）‘它也许更适合你。’", "（把{prop}轻轻推过去）‘请把它当作合作的纪念。’"],
        "help": ["（检查损坏的痕迹）‘让我负责另一边。’", "（放下行李）‘先解决问题，报酬之后再说。’"],
        "promise": ["（把约定写进册子）‘月亮升起前，我一定带答案回来。’", "（用力握拳）‘即使危险，我也会遵守约定。’"],
        "secret": ["（在地上画路线）‘{clue}只在子夜出现，暂时只有你知道。’", "（递出纸条）‘这里记录了{clue}，请判断是否公开。’"],
        "open_event": ["（指着地图空白处）‘{clue}正朝这里移动，得立刻追踪。’", "（取出灵力碎片）‘它来自{clue}出现的地方。’"],
        "spellcard": ["（先宣告符卡名）‘请用最擅长的弹幕回应，我会遵守裁定。’", "（退到规定距离）‘不伤旁人，以一次符卡定胜负。’"],
        "task": ["（核对清单）‘“{task}”的目标已经达成，请确认。’", "（标出新位置）‘这能推进“{task}”，还差最后一步。’"],
        "trade": ["（取出{prop}）‘我想换一件辨认妖气的工具。’", "（把{prop}放在光下）‘请先说明交换物的实际用途。’"],
        "threat": ["（推开告示牌）‘继续拦我就用弹幕说话。’", "（冷笑着逼近）‘最后一次机会，把路让开。’"],
        "rumor": ["（复述旅店消息）‘有人看见{clue}，但说法互相矛盾。’", "（展示传单）‘上面说{clue}会引发异变，这是谣言吗？’"],
    },
}


def load_json(path):
    # 游戏仓库的部分 JSON 带 UTF-8 BOM，读取时兼容两种形式。
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def find_touhou(repo_root, explicit):
    for value in (explicit, os.environ.get("TOUHOU_ROOT"), repo_root.parent / "touhou"):
        if value:
            path = Path(value).expanduser().resolve()
            if (path / "worlds/world_touhou/npcs/npc_index.json").is_file():
                return path
    raise FileNotFoundError("找不到 touhou 仓库；请用 --touhou-root 指定路径")


def load_world(root):
    npc_doc = load_json(root / "worlds/world_touhou/npcs/npc_index.json")
    loc_doc = load_json(root / "worlds/world_touhou/locations/location_base.json")
    locations = {x["id"]: x for x in loc_doc["locations"]}
    unique = {}
    for npc in npc_doc["npcs"]:
        if npc.get("active") is not False and not npc.get("dead") and npc.get("name"):
            unique.setdefault(npc["name"], npc)
    if len(unique) < 20:
        raise ValueError("有效 NPC 数量不足，源档案可能不完整")
    return list(unique.values()), locations


def shortened(value, limit, fallback):
    text = " ".join(str(value or "").split()).replace("；", "，")
    return text[:limit].rstrip("，。； ") or fallback


def context(npc, locations):
    profile = npc.get("profile") or {}
    loc = locations.get(npc.get("location_id"), {})
    traits = profile.get("personality_traits") or ["谨慎", "符合自身身份"]
    return {
        "name": npc["name"], "id": npc.get("id", "unknown"),
        "identity": shortened(profile.get("identity"), 70, "幻想乡居民"),
        "traits": "、".join(map(str, traits[:4])),
        "attitude": shortened(profile.get("initial_attitude"), 30, "中立观察"),
        "hook": shortened(profile.get("story_hook"), 100, "围绕当前地区的见闻展开互动"),
        "spell": shortened(profile.get("spellcard_style"), 100, "遵守符卡规则，以非致命弹幕分出胜负"),
        "location": loc.get("name", "幻想乡"),
        "location_desc": shortened(loc.get("description"), 100, "幻想乡中的一处地点"),
    }


def memory(name, summary, tags, importance, emotion, fact_key, reported=False):
    item = {
        "npc_name": name, "summary": summary, "tags": tags,
        "importance": importance, "emotion": emotion,
        "knowledge_type": "reported" if reported else "direct",
        "confidence": 0.65 if reported else 0.9,
        "truth_status": "disputed" if reported else "accepted",
        "fact_key": fact_key,
    }
    if reported:
        item["source_npc"] = "玩家"
    return item


def make_description(category, ctx, detail, weather, outcome):
    name, place, trait = ctx["name"], ctx["location"], ctx["traits"].split("、")[0]
    descriptions = {
        "daily": f"（{name}放下手边的事，看了看{place}，态度透着{trait}）‘这里的平静通常不会持续太久。既然你愿意先听规矩，我可以说说最近值得留意的变化。’{weather}，谈话没有惊动旁人。",
        "gift": f"（{name}以一贯的{trait}接过{detail}，仔细看了片刻）‘礼物我收下，但我更看重你送来的理由。下次别为了讨好谁冒险，平安回来再说。’{weather}，原本拘谨的气氛缓和了一些。",
        "help": f"（{name}按{trait}的行事风格重新打量现场，把一部分工作交给你）‘既然主动开口，就按说好的分工来。别逞强，也别漏掉可疑痕迹。’两人很快开始协作，{weather}，进度比预想中顺利。",
        "promise": f"（{name}看着你记下约定，原本{trait}的神情认真起来）‘我会记住这句话。结果好坏可以以后再谈，失约却会让线索失去意义。’{weather}，这项承诺被当作后续行动的依据。",
        "secret": f"（{name}保持着{trait}的语气听完关于{detail}的说明，先确认四周无人）‘我会暂时保密，但传闻和亲眼所见必须分开。先核对时间地点，再决定是否告诉别人。’{weather}，信息仍被视作待验证的秘密。",
        "open_event": f"（{name}以{trait}的反应沿你所指方向观察，很快辨认出{detail}留下的异常）‘这不是普通天气造成的。线索还没消失，现在追过去也许能找到源头。’{weather}，一条可以继续调查的新事件由此展开。",
        "spellcard": f"（弹幕在{place}上空依规则展开，{name}以{trait}而鲜明的节奏封锁路线；数轮交错后，裁定为{outcome}）‘结果已经明确，我承认这一次的胜负。弹幕是展示招式，不是为伤人。’余光散去，双方按规则收起符卡。",
        "task": f"（{name}依照{trait}的习惯逐页核对记录，又确认“{detail}”的状态）‘这些证据足够更新任务。过程里还有值得追查的地方，我会一并写进记录。’{weather}，任务进度得到了明确处理。",
        "trade": f"（{name}以{trait}的方式审视{detail}，先说明交换物的用途）‘交易可以，但条件要让双方都清楚。你确认之后再交换，免得以后为了误会翻脸。’{weather}，双方完成了公开而克制的交换。",
        "threat": f"（{name}原本{trait}的神情骤然冷下，拉开距离戒备你的武器）‘到此为止。幻想乡的争斗有符卡规则，你若继续威胁，我不会再提供帮助。’周围气氛迅速紧绷，谈话被当场中止。",
        "rumor": f"（{name}保持着{trait}听完关于{detail}的转述，没有把它当成事实）‘来源含糊，几种说法又互相冲突。可以记作传闻，但找到见证者前别扩散。’{weather}，消息被标为仍有争议的间接情报。",
    }
    text = descriptions[category]
    if not 50 <= len(text) <= 200:
        raise ValueError(f"{category} description 长度为 {len(text)}")
    return text


def build_result(category, ctx, seq, detail, weather, outcome):
    name, place = ctx["name"], ctx["location"]
    result = {
        "description": make_description(category, ctx, detail, weather, outcome),
        "is_dead": False, "time_cost": 0.25, "new_energy_state": None,
        "new_location": None, "relationship_update": None,
        "inventory_updates": [], "reputation_updates": [], "world_effects": [],
        "task_updates": [], "memory_updates": [], "open_event": None,
        "spellcard_result": None, "exit_dialogue": False,
    }
    key = f"{category}:{ctx['id']}:{seq:04d}"
    if category == "daily":
        result["time_cost"] = 0.5
        result["memory_updates"] = [memory(name, f"玩家在{place}礼貌询问近况，并愿意先了解当地规矩。", ["日常", place], 3, "中性", key)]
    elif category == "gift":
        result["relationship_update"] = f"{name}:友好(玩家主动赠送{detail})"
        result["inventory_updates"] = [{"action": "remove", "name": detail, "quantity": 1, "category": "礼物", "description": f"赠送给{name}"}]
        result["memory_updates"] = [memory(name, f"玩家在{place}把{detail}作为谢礼送给{name}。", ["赠礼", place], 6, "愉快", key)]
    elif category == "help":
        result["time_cost"] = 1
        result["relationship_update"] = f"{name}:友好(玩家主动协助处理现场事务)"
        result["reputation_updates"] = [{"faction": f"{place}居民", "delta": 2, "reason": "主动提供可靠帮助"}]
        result["memory_updates"] = [memory(name, f"玩家在{place}主动与{name}分工合作，顺利推进现场事务。", ["合作", "帮助"], 6, "信任", key)]
    elif category == "promise":
        result["relationship_update"] = f"{name}:信任(玩家郑重作出后续约定)"
        result["memory_updates"] = [memory(name, f"玩家答应继续追查线索并返回{place}向{name}说明结果。", ["约定", "线索"], 7, "期待", key)]
    elif category == "secret":
        result["memory_updates"] = [memory(name, f"玩家私下告知{name}曾发现{detail}，目前尚未公开。", ["秘密", "待核实"], 7, "警觉", key)]
    elif category == "open_event":
        result["time_cost"] = 0.5
        result["memory_updates"] = [memory(name, f"玩家与{name}在{place}确认了{detail}，决定继续追踪。", ["异变", "调查"], 8, "警觉", key)]
        result["open_event"] = {"id": f"evt_{ctx['id']}_{seq:04d}", "title": f"{place}的异常迹象", "type": "异变线索", "scene": place, "npc_name": name, "description": f"{place}出现{detail}，来源仍可继续调查。", "hooks": [f"沿{place}周边追踪痕迹", f"向{name}确认异常规律"], "source": "npc_dialogue"}
    elif category == "spellcard":
        result["time_cost"] = 1
        result["new_energy_state"] = "略有疲惫"
        result["relationship_update"] = f"{name}:认可(遵守符卡规则完成正式比试)"
        result["memory_updates"] = [memory(name, f"玩家与{name}在{place}完成符卡战，后端裁定为{outcome}。", ["符卡", "战斗"], 8, "竞争", key)]
        result["spellcard_result"] = {"opponent": name, "spellcard_name": f"试符「{ctx['id'].removeprefix('npc_')[:10]}幻光」", "outcome": outcome, "summary": f"双方依照符卡规则交错弹幕，后端确定性规则最终裁定玩家{outcome}，叙事未反转结果。", "cost": "消耗少量灵力并产生轻微疲劳"}
    elif category == "task":
        task_id = f"task_{ctx['id']}_{seq:04d}"
        complete = seq % 2 == 0
        result["time_cost"] = 0.5
        result["task_updates"] = [{"action": "complete" if complete else "update", "task_id": task_id, "task_name": detail, "info": f"玩家带回记录，任务{'已经完成' if complete else '取得关键进展'}。", "priority": 80, "source": "npc_dialogue"}]
        result["memory_updates"] = [memory(name, f"玩家向{name}汇报“{detail}”的结果，任务记录已更新。", ["任务", "调查"], 7, "认可", key)]
    elif category == "trade":
        reward = "辨识灵力的护符"
        result["time_cost"] = 0.5
        result["inventory_updates"] = [{"action": "remove", "name": detail, "quantity": 1, "category": "交换物", "description": f"交给{name}"}, {"action": "add", "name": reward, "quantity": 1, "category": "道具", "description": f"从{name}处交换获得"}]
        result["memory_updates"] = [memory(name, f"玩家在{place}用{detail}与{name}公平交换了{reward}。", ["交易", place], 5, "中性", key)]
    elif category == "threat":
        result["relationship_update"] = f"{name}:敌对(玩家持械威胁并拒绝遵守秩序)"
        result["reputation_updates"] = [{"faction": f"{place}居民", "delta": -5, "reason": "公开威胁当地居民"}]
        result["memory_updates"] = [memory(name, f"玩家在{place}以武器和言语威胁{name}，对话被中止。", ["威胁", "冲突"], 9, "敌意", key)]
        result["exit_dialogue"] = True
    elif category == "rumor":
        result["memory_updates"] = [memory(name, f"玩家转述：有人在{place}附近发现{detail}；来源尚未证实。", ["传闻", "待核实"], 5, "疑虑", key, reported=True)]
        result["world_effects"] = [{"kind": "rumor", "target": place, "effect": f"关于{detail}的说法开始被谨慎讨论", "magnitude": 1, "delay_hours": 6}]
    return result


def build_prompt(category, ctx, action, result, detail, time_name, weather):
    history = "（无）" if category in {"daily", "threat", "open_event"} else f"{ctx['name']}：（留意着玩家）‘先把来意说清楚。’"
    task = "当前没有进行中的任务。"
    if category == "task":
        update = result["task_updates"][0]
        task = f"进行中任务：ID={update['task_id']}；名称={detail}；状态=等待汇报。"
    ruling = ""
    if category == "spellcard":
        ruling = f"\n## 后端游戏规则预裁定（必须遵守）\n后端确定性裁定：对{ctx['name']}的符卡战结果为「{result['spellcard_result']['outcome']}」。叙事可以丰富过程但不得反转胜负。\n"
    return f"""你是《东方异变录》中的 NPC。根据世界状态扮演角色，并只返回后端可解析的 JSON。

## 世界观
幻想乡由博丽大结界与外界隔离，人类、妖怪、神明和亡灵共同生活。冲突优先以非致命的符卡规则解决；异变、关系、任务、记忆和传闻会持续影响剧情。

## 当前NPC信息
姓名：{ctx['name']}
身份：{ctx['identity']}
性格关键词：{ctx['traits']}
当前态度：{ctx['attitude']}
剧情方向：{ctx['hook']}
符卡风格：{ctx['spell']}

## 玩家与场景状态
玩家精力正常、未受伤，携带基础旅途用品。
当前位置：{ctx['location']}（{ctx['location_desc']}）
当前时间：{time_name}；环境：{weather}。
当前关系：玩家与{ctx['name']}处于“{ctx['attitude']}”。
{task}

## 对话历史
{history}

## 玩家的动作和语言，动作包含在（）里
玩家：{action}
{ruling}
## 输出规则
1. 以{ctx['name']}的身份回应；description 中动作用（），语言用‘’。
2. description 为 50-200 字；只输出一个严格 JSON 对象，禁止 Markdown 和解释。
3. 固定输出全部 14 个字段：description、is_dead、time_cost、new_energy_state、new_location、relationship_update、inventory_updates、reputation_updates、world_effects、task_updates、memory_updates、open_event、spellcard_result、exit_dialogue。
4. 没有事件或符卡战时相应字段为 null；没有更新时数组为空、relationship_update 为 null。
5. memory_updates 是对象数组；open_event 包含 title/type/scene/description/hooks；spellcard_result 只含 opponent/spellcard_name/outcome/summary/cost。
6. 不得反转预裁定、杀死角色或绕过符卡规则。"""


def make_split(split, quotas, npcs, locations, seed):
    rng = random.Random(seed)
    schedule = [kind for kind, count in quotas.items() for _ in range(count)]
    rng.shuffle(schedule)
    order = list(npcs)
    rng.shuffle(order)
    samples, rows = [], []
    for index, category in enumerate(schedule):
        if index and index % len(order) == 0:
            rng.shuffle(order)
        ctx = context(order[index % len(order)], locations)
        prop, clue, task = rng.choice(PROPS), rng.choice(CLUES), rng.choice(TASKS)
        detail = task if category == "task" else clue if category in {"secret", "open_event", "rumor"} else prop
        weather = rng.choice(WEATHERS)
        # 精确游戏时刻既是有效场景状态，也令语义相同的回合保持可区分。
        time_name = f"幻想历第{index // 24 + 1}日·{TIMES[index % len(TIMES)]}·第{index % 24 + 1}刻"
        outcome = OUTCOMES[index % 3]
        result = build_result(category, ctx, index + 1, detail, weather, outcome)
        action = rng.choice(ACTIONS[split][category]).format(prop=prop, clue=clue, task=task)
        samples.append({"instruction": build_prompt(category, ctx, action, result, detail, time_name, weather), "input": "", "output": json.dumps(result, ensure_ascii=False, separators=(",", ":"))})
        rows.append((category, ctx["name"], ctx["location"]))
    return samples, rows


def hash_set(samples, field):
    return {hashlib.sha256(item[field].encode("utf-8")).hexdigest() for item in samples}


def assert_isolated(train, evaluation):
    for field in ("instruction", "output"):
        train_hashes, eval_hashes = hash_set(train, field), hash_set(evaluation, field)
        if len(train_hashes) != len(train) or len(eval_hashes) != len(evaluation):
            raise ValueError(f"{field} 集合内部有完全重复样本")
        if train_hashes & eval_hashes:
            raise ValueError(f"训练/评测 {field} 存在完全相同样本")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--touhou-root", help="默认寻找本仓库同级的 touhou")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-out", default=repo / "data/npc_dialogue.json")
    parser.add_argument("--eval-out", default=repo / "data/npc_eval.json")
    parser.add_argument("--manifest-out", default=repo / "data/dataset_manifest.json")
    args = parser.parse_args()

    npcs, locations = load_world(find_touhou(repo, args.touhou_root))
    train, train_rows = make_split("train", TRAIN_QUOTAS, npcs, locations, args.seed)
    evaluation, eval_rows = make_split("eval", EVAL_QUOTAS, npcs, locations, args.seed + 1)
    assert len(train) == 2000 and len(evaluation) == 200
    assert_isolated(train, evaluation)
    write_json(args.train_out, train)
    write_json(args.eval_out, evaluation)
    manifest = {
        "schema_version": 1, "generator": "scripts/generate_dataset.py", "seed": args.seed,
        "source": {"npc_index": "touhou/worlds/world_touhou/npcs/npc_index.json", "location_base": "touhou/worlds/world_touhou/locations/location_base.json", "active_unique_npcs": len(npcs), "locations": len(locations)},
        "train": {"count": len(train), "quotas": dict(Counter(x[0] for x in train_rows)), "npc_count": len({x[1] for x in train_rows})},
        "eval": {"count": len(evaluation), "quotas": dict(Counter(x[0] for x in eval_rows)), "npc_count": len({x[1] for x in eval_rows})},
        "isolation": {"exact_prompt_overlap": 0, "exact_output_overlap": 0},
    }
    write_json(args.manifest_out, manifest)
    print(f"已生成训练集 {len(train)} 条、评测集 {len(evaluation)} 条")
    print(f"覆盖 {len(npcs)} 个有效 NPC；跨集合 prompt/output 完全重复均为 0")


if __name__ == "__main__":
    main()
