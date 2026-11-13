import json
from PIL import Image
from scripts.make_toy_data import generate, make_scene
import random

def test_modes_and_labels(tmp_path):
    for mode in ("easy", "mixed", "hard"):
        root = generate(tmp_path/mode, train=4, val=2, mode=mode)
        for split,n in (("train",4),("val",2)):
            rows = [json.loads(s) for s in (root/f"{split}.jsonl").read_text().splitlines()]
            assert len(rows) == n
            for row in rows:
                target = [o for o in row["objects"] if
                          row["expression"] == f'the {o["color"]} {o["shape"]}']
                assert len(target) == 1
                assert row["bbox_xyxy"] == target[0]["bbox_xyxy"]
                assert row["target_visible_fraction"] >= 0.68
                assert Image.open(root/row["image"]).size == (224,224)

def test_hard_distractors():
    _, row = make_scene(random.Random(123), "hard")
    objects = row["objects"]
    assert len(objects) >= 5
    assert len({(o["color"],o["shape"]) for o in objects}) == len(objects)
    assert any(o["color"] == objects[0]["color"] and o != objects[0] for o in objects)
