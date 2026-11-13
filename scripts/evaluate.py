import argparse,json
import torch
from torch.utils.data import DataLoader
from grounding.data import ReferringDataset,collate
from grounding.engine import evaluate
from grounding.checkpoints import load_grounder

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/toy/val.jsonl")
    p.add_argument("--image-root",default="data/toy")
    p.add_argument("--checkpoint",default="checkpoints/grounder.pt")
    p.add_argument("--batch-size",type=int,default=4)
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,ck=load_grounder(a.checkpoint,device)
    val=DataLoader(ReferringDataset(a.data,a.image_root),
                   batch_size=a.batch_size,collate_fn=collate,num_workers=0)
    print(json.dumps({"arch":ck.get("arch","baseline"),**evaluate(model,val,device)},indent=2))

if __name__=="__main__":
    main()
