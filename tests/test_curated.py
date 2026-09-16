"""Offline regression tests; no model loading and no API calls."""
import ast
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_curated_dataset as builder


class CuratedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs, cls.manifest = builder.build()
        cls.records = builder.load_authored()
        cls.profiles = {p["name"]: p for path in (ROOT / "data").glob("canon_profiles*.json")
                        for p in builder.read(path)["profiles"]}

    def test_checked_in_files_rebuild_exactly(self):
        for path, value in self.outputs.items():
            self.assertEqual(path.read_text(encoding="utf-8"), builder.serialized(value), str(path))

    def test_progress_is_explicit_and_consistent(self):
        self.assertFalse(self.manifest["human_review_completed"])
        for split, total in (("train", 3000), ("eval", 200)):
            progress = self.manifest["progress"][split]
            self.assertEqual(progress["total"], total)
            self.assertEqual(progress["rewritten"], sum(r["split"] == split for r in self.records))
            self.assertEqual(progress["pending"], sum(progress["pending_by_npc"].values()))
            self.assertEqual(progress["rewritten"] + progress["pending"], total)

    def test_descriptions_and_deltas_are_verbatim_not_generated(self):
        for record in self.records:
            sample = builder.assemble(record, self.profiles[record["npc"]])
            answer = json.loads(sample["output"])
            self.assertEqual(answer["description"], record["description"])
            self.assertIn(record["context"], sample["instruction"])
            self.assertIn(record["player"], sample["instruction"])
            for field, value in record["delta"].items():
                self.assertEqual(answer[field], value)

    def test_sources_are_revision_pinned_and_not_full_page_copies(self):
        for record in self.manifest["records"]:
            source = self.manifest["sources"][record["source_title"]]
            self.assertGreater(source["revision"], 0)
            self.assertIn("oldid=" + str(source["revision"]), source["url"])
            self.assertEqual(len(source["sha256"]), 64)
            self.assertNotIn("text", source)

    def test_selected_npc_slots_are_not_overfilled(self):
        for split, filename in builder.FILES.items():
            full = self.outputs[ROOT / "data" / (filename + ".json")]
            full_counts = Counter(builder.npc_name(s) for s in full)
            authored_counts = Counter(r["npc"] for r in self.records if r["split"] == split)
            for name, count in authored_counts.items():
                self.assertGreaterEqual(full_counts[name], count)

    def test_eval_has_positive_and_negative_state_cases(self):
        evaluation = [r for r in self.records if r["split"] == "eval"]
        self.assertTrue(any(not r["delta"] for r in evaluation))
        for field in ("memory_updates", "spellcard_result", "open_event", "task_updates", "inventory_updates"):
            self.assertTrue(any(r["delta"].get(field) for r in evaluation), field)

    def test_duplicate_description_rejected_even_with_different_state(self):
        records = copy.deepcopy(self.records)
        records[1]["description"] = records[0]["description"]
        reviewed = {split: [builder.assemble(r, self.profiles[r["npc"]]) for r in records if r["split"] == split]
                    for split in builder.FILES}
        with self.assertRaises(ValueError):
            builder.audit(records, reviewed)

    def test_old_battle_cannot_create_new_result(self):
        records = copy.deepcopy(self.records)
        record = next(r for r in records if r["case"] == "old_battle")
        record["delta"] = {"exit_dialogue": True}
        with self.assertRaisesRegex(ValueError, "Recall-only"):
            builder.audit(records, {"train": [], "eval": []})

    def test_default_entrypoints_use_rewritten_data(self):
        for filename in ("train_sft.py", "eval/eval.py"):
            source = (ROOT / filename).read_text(encoding="utf-8")
            ast.parse(source)
            self.assertIn("npc_eval_reviewed.json", source)
        evaluation = (ROOT / "eval/eval.py").read_text(encoding="utf-8")
        self.assertIn("if args.paid_judge:", evaluation)
        self.assertNotIn("if not args.skip_judge:", evaluation)
        self.assertLess(evaluation.index("    save(rows)"), evaluation.index("    if args.paid_judge:"))
        self.assertIn('raw["input_ids"] if hasattr(raw, "keys") else raw', evaluation)
        for filename in ("train_npc_qlora.yaml", "train_npc_qlora_smoke.yaml", "train_npc_qlora_7b.yaml"):
            config = (ROOT / "configs" / filename).read_text(encoding="utf-8")
            self.assertIn("dataset: npc_dialogue_reviewed", config)
            if filename != "train_npc_qlora_smoke.yaml":
                self.assertIn("eval_dataset: npc_eval_reviewed", config)

    def test_aliases_are_included_in_similarity_audit(self):
        records = [dict(id="train/alias/a", split="train", npc="alias", case="a",
                        context="one", player="one", description="abcdefghij", delta={}),
                   dict(id="eval/canonical/b", split="eval", npc="canonical", case="b",
                        context="two", player="two", description="abcdefghik", delta={})]
        with self.assertRaisesRegex(ValueError, "Near-duplicate"):
            builder.audit(records, {}, {"alias": "same", "canonical": "same"})

    def test_requested_milestone_and_legacy_mapping_are_explicit(self):
        target = builder.read(ROOT / "data/rewrite_target.json")
        self.assertEqual(self.manifest["progress"]["train"]["rewritten"], target["requested_total_train"])
        self.assertFalse(target["paid_api_allowed"])
        self.assertEqual(self.manifest["progress"]["eval"]["rewritten"], target["requested_total_eval"])
        self.assertEqual(self.manifest["progress"]["eval"]["pending_by_npc"], {})
        resolution = self.manifest["identity_resolution"]
        self.assertEqual(resolution["project_id_inferred_records"], 4)
        self.assertEqual(resolution["other_source_backed_records"], 1196)
        self.assertEqual(target["identity_inferred_eval_records"], 4)
        mapping = builder.read(ROOT / "data/legacy_identity_mapping.json")
        self.assertEqual({r["legacy_label"] for r in mapping["mappings"]}, {"天独铭", "天肱"})
        for row in mapping["mappings"]:
            self.assertEqual(self.profiles[row["legacy_label"]]["source_title"], row["source_title"])


if __name__ == "__main__":
    unittest.main()
