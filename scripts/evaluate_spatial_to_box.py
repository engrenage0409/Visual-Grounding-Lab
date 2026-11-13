"""Evaluate V7.2 head on held-out data; report frozen V6 prediction too."""
import argparse,json
import torch
from torch.utils.data import DataLoader
from grounding.data import ReferringDataset,collate
from grounding.spatial_to_box_checkpoints import load_spatial_to_box
from scripts.train_spatial_to_box import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--v6-checkpoint",default=None,help="Override V6 path from saved checkpoint")
    p.add_argument("--data",default="data/v4_hard/val.jsonl")
    p.add_argument("--image-root",default="data/v4_hard")
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--constrain-box",action="store_true")
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,ck=load_spatial_to_box(a.checkpoint,device,a.v6_checkpoint)
    loader=DataLoader(ReferringDataset(a.data,a.image_root),batch_size=a.batch_size,
                      shuffle=False,collate_fn=collate,num_workers=0)
    print(json.dumps({"arch":ck["arch"],"checkpoint_epoch":ck["epoch"],
                      **evaluate(model,loader,device,a.constrain_box)},indent=2))


if __name__=="__main__":
    main()
