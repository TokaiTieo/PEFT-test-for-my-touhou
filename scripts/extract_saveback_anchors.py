"""从 touhousaveback 的最新测试存档提取、匿名化并去重剧情锚点。

输出只保留生成训练数据所需的 NPC、场景、玩家动作、叙事片段和状态事实；
角色 UUID、玩家姓名、高权限 prompt、时间戳等不写入派生文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

BANNED = ("成人向", "做爱", "性交", "强奸", "裸体", "性器", "高潮", "高权限叙事模式")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def clean(text, player_names):
    value = str(text or "")
    for name in sorted(player_names, key=len, reverse=True):
        if name:
            value = value.replace(name, "玩家")
    value = re.sub(r"\n*⏰\s*过了[^\n]*", "", value)
    value = value.replace("「", "‘").replace("」", "’").replace("“", "‘").replace("”", "’")
    value = re.sub(r"'([^']+)'", r"‘\1’", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clipped(text, limit=190):
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    head = value[:limit]
    stops = [head.rfind(mark) for mark in "。！？；’）"]
    stop = max(stops)
    return head[: stop + 1] if stop >= 60 else head.rstrip("，、：； ") + "。"


def description_from(raw, npc_name, player_names):
    paragraphs = [clean(part, player_names) for part in re.split(r"\n\s*\n", str(raw or ""))]
    paragraphs = [part for part in paragraphs if part and not any(word in part for word in BANNED)]
    relevant_indexes = [index for index, part in enumerate(paragraphs) if npc_name in part]
    relevant = [paragraphs[index] for index in relevant_indexes]
    text = clipped(" ".join(relevant[:2] or paragraphs[:1]), 185)
    if not text:
        return ""
    start = relevant_indexes[0] if relevant_indexes else 0
    nearby = " ".join(paragraphs[start : start + 3])
    speeches = re.findall(r"‘([^’]{2,})’", nearby)
    if speeches:
        action = re.sub(r"‘[^’]*’", "", relevant[0] if relevant else paragraphs[0]).strip(" （），。；")
        text = f"（{clipped(action, 72)}）‘{clipped(max(speeches, key=len), 105)}’"
    elif "‘" in text:
        before, after = text.split("‘", 1)
        before = before.strip(" ，。；")
        if before and not before.startswith("（"):
            text = f"（{clipped(before, 70)}）‘{after}"
    elif not text.startswith("（"):
        text = f"（{text.rstrip('。')}）"
    if len(text) < 50:
        text += f"{npc_name}把这一刻记在心里，并等待玩家说明下一步打算。"
    return clipped(text, 200)


def first_mentioned(text, names):
    hits = [(str(text).find(name), name) for name in names if name in str(text)]
    return min(hits)[1] if hits else None


def dedupe(items, fields):
    result, seen = [], set()
    for item in items:
        key = tuple(json.dumps(item.get(field), ensure_ascii=False, sort_keys=True) for field in fields)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-root", default=repo.parent / "touhousaveback")
    parser.add_argument("--touhou-root", default=repo.parent / "touhou")
    parser.add_argument("--out", default=repo / "data/saveback_anchors.json")
    args = parser.parse_args()

    save_dir = Path(args.save_root) / "sessions/characters"
    files = sorted(p for p in save_dir.glob("*.json") if not p.name.endswith("_tasks.json"))
    saves = [read_json(path) for path in files]
    if not saves:
        raise FileNotFoundError(f"{save_dir} 下没有主存档 JSON")
    player_names = {str(save.get("profile", {}).get("name") or "") for save in saves}
    npc_doc = read_json(Path(args.touhou_root) / "worlds/world_touhou/npcs/npc_index.json")
    npc_rows = {item["name"]: item for item in npc_doc["npcs"] if item.get("name")}
    npc_names = sorted(npc_rows, key=len, reverse=True)

    conversations, memories, events, spells, relationships = [], [], [], [], []
    for save in saves:
        history = save.get("conversation_history") or []
        for index, turn in enumerate(history[:-1]):
            answer = history[index + 1]
            if turn.get("speaker") in {"旁白", "系统"} or answer.get("speaker") != "旁白":
                continue
            npc_name = first_mentioned(answer.get("content"), npc_names)
            if not npc_name:
                continue
            action = clipped(clean(turn.get("content"), player_names), 180)
            desc = description_from(answer.get("content"), npc_name, player_names)
            if not action or not 50 <= len(desc) <= 200 or any(word in action + desc for word in BANNED):
                continue
            previous = history[index - 1].get("content", "") if index else ""
            source_hash = hashlib.sha256((action + "\n" + desc).encode("utf-8")).hexdigest()[:16]
            conversations.append({
                "source_hash": source_hash,
                "npc_name": npc_name,
                "scene": clean(answer.get("scene") or turn.get("scene") or "幻想乡", player_names),
                "player_action": action,
                "history": clipped(clean(previous, player_names), 240),
                "description": desc,
            })

        for npc_name, rows in (save.get("npc_memories") or {}).items():
            if npc_name not in npc_rows:
                continue
            for row in rows or []:
                summary = clipped(clean(row.get("summary"), player_names), 260)
                if summary and not any(word in summary for word in BANNED):
                    memories.append({"npc_name": npc_name, "summary": summary, "tags": [clean(x, player_names) for x in (row.get("tags") or [])[:5]], "importance": max(1, min(10, int(row.get("importance") or 5))), "emotion": clean(row.get("emotion") or "中性", player_names)})

        for row in save.get("open_events") or []:
            combined = f"{row.get('title', '')} {row.get('description', '')} {' '.join(row.get('hooks') or [])}"
            npc_name = first_mentioned(combined, npc_names)
            if not npc_name:
                continue
            event = {key: clean(row.get(key), player_names) for key in ("title", "type", "scene", "description")}
            event["npc_name"] = npc_name
            event["hooks"] = [clean(x, player_names) for x in (row.get("hooks") or []) if clean(x, player_names)][:4]
            if event["title"] and event["description"] and event["hooks"] and not any(word in json.dumps(event, ensure_ascii=False) for word in BANNED):
                events.append(event)

        for row in save.get("spellcard_history") or []:
            opponent = first_mentioned(row.get("opponent"), npc_names)
            if not opponent:
                continue
            spell = {key: clean(row.get(key), player_names) for key in ("scene", "spellcard_name", "outcome", "summary", "cost")}
            spell["opponent"] = opponent
            if spell["spellcard_name"] and spell["summary"] and not any(word in json.dumps(spell, ensure_ascii=False) for word in BANNED):
                spells.append(spell)

        relation_map = save.get("relationships_map") or save.get("relationships") or {}
        for npc_name, state in relation_map.items():
            if npc_name in npc_rows and state:
                relationships.append({"npc_name": npc_name, "state": clipped(clean(state, player_names), 180)})

    conversations = dedupe(conversations, ("player_action", "description"))
    memories = dedupe(memories, ("npc_name", "summary"))
    events = dedupe(events, ("title", "description"))
    spells = dedupe(spells, ("opponent", "spellcard_name", "summary"))
    relationships = dedupe(relationships, ("npc_name", "state"))
    payload = {
        "schema_version": 1,
        "provenance": "touhousaveback/sessions/characters 的最新主存档；已匿名化、去重和安全过滤",
        "source_save_count": len(saves),
        "counts": {"conversations": len(conversations), "memories": len(memories), "events": len(events), "spells": len(spells), "relationships": len(relationships)},
        "conversations": conversations,
        "memories": memories,
        "events": events,
        "spells": spells,
        "relationships": relationships,
    }
    write_json(args.out, payload)
    print(f"已从 {len(saves)} 个最新主存档提取锚点：{payload['counts']}")
    print(f"输出：{args.out}")


if __name__ == "__main__":
    main()
