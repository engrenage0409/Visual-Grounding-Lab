import argparse,json,random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from grounding.data import ReferringDataset,collate
from grounding.engine import grounding_loss,evaluate
from grounding.model import CLIPGrounder

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",default="data/toy/train.jsonl")
    p.add_argument("--val",default="data/toy/val.jsonl")
    p.add_argument("--image-root",default="data/toy")
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--out",default="checkpoints/grounder.pt")
    p.add_argument("--model",default="openai/clip-vit-base-patch32")
    a=p.parse_args()
    torch.manual_seed(42);random.seed(42)
    device="cuda" if torch.cuda.is_available() else "cpu"
    train=DataLoader(ReferringDataset(a.train,a.image_root),batch_size=a.batch_size,
                     shuffle=True,collate_fn=collate,num_workers=0)
    val=DataLoader(ReferringDataset(a.val,a.image_root),batch_size=a.batch_size,
                   shuffle=False,collate_fn=collate,num_workers=0)
    model=CLIPGrounder(a.model).to(device)
    opt=torch.optim.AdamW((v for v in model.parameters() if v.requires_grad),lr=a.lr)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    best=-1.
    for epoch in range(1,a.epochs+1):
        model.train();losses=[]
        for batch in tqdm(train,desc=f"Epoch {epoch}"):
            opt.zero_grad(set_to_none=True)
            pred=model(batch["images"],batch["texts"])
            loss,_=grounding_loss(pred,batch["boxes"].to(device))
            loss.backward();opt.step();losses.append(loss.item())
        metrics=evaluate(model,val,device)
        print(json.dumps({"epoch":epoch,"loss":sum(losses)/len(losses),**metrics}))
        if metrics["acc_at_0.5"]>best:
            best=metrics["acc_at_0.5"]
            head={k:v for k,v in model.state_dict().items() if not k.startswith("clip.")}
            torch.save({"head":head,"model_name":a.model,"metrics":metrics,"epoch":epoch},path)
    print("Saved:",path)

if __name__=="__main__":
    main()
