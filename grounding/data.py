import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset

class ReferringDataset(Dataset):
    """JSONL fields: image, expression, bbox_xyxy (original pixel coordinates)."""
    def __init__(self,jsonl,image_root):
        self.root=Path(image_root)
        with Path(jsonl).open(encoding="utf-8") as f:
            self.rows=[json.loads(line) for line in f if line.strip()]
        if not self.rows:
            raise ValueError("Empty dataset")
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        path=Path(row["image"])
        if not path.is_absolute():
            path=self.root/path
        with Image.open(path) as im:
            im=im.convert("RGB")
            w,h=im.size
            image=im.copy()
        box=torch.tensor(row["bbox_xyxy"],dtype=torch.float32)
        if box.numel()!=4 or not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
            raise ValueError(f"Invalid box: {path}")
        box=box/torch.tensor([w,h,w,h],dtype=torch.float32)
        return {"image":image,"text":row["expression"],"box":box,"path":str(path)}
def collate(batch):
    return {"images":[v["image"] for v in batch],
            "texts":[v["text"] for v in batch],
            "boxes":torch.stack([v["box"] for v in batch]),
            "paths":[v["path"] for v in batch]}
