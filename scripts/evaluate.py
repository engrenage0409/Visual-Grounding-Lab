import argparse,json
import torch
from torch.utils.data import DataLoader
from grounding.data import ReferringDataset,collate
from grounding.engine import evaluate
from grounding.model import CLIPGrounder

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/toy/val.jsonl")
    p.add_argument("--image-root",default="data/toy")
    p.add_argument("--checkpoint",default="checkpoints/grounder.pt")
    p.add_argument("--batch-size",type=int,default=4)
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    model=CLIPGrounder(ck["model_name"]).to(device)
    model.load_state_dict(ck["head"],strict=False)
    val=DataLoader(ReferringDataset(a.data,a.image_root),
                   batch_size=a.batch_size,collate_fn=collate,num_workers=0)
    print(json.dumps(evaluate(model,val,device),indent=2))

if __name__=="__main__":
    main()
