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
    p.add_argument("--arch",choices=["baseline","spatial","region","region_af","coarse_fine"],default="spatial")
    p.add_argument("--loss",choices=["iou","giou"],default="giou")
    p.add_argument("--spatial-weight",type=float,default=1.0,
                   help="Spatial CE weight for spatial arch; selection CE weight for region; classification/quality weight for region_af")
    p.add_argument("--init-checkpoint",default=None,help="For V6: load trained V4 spatial head before refinement training")
    p.add_argument("--freeze-coarse",action="store_true",help="Train only V6 refinement head; requires --arch coarse_fine and --init-checkpoint")
    a=p.parse_args()
    if a.freeze_coarse and (a.arch!="coarse_fine" or not a.init_checkpoint):
        p.error("--freeze-coarse requires --arch coarse_fine and --init-checkpoint")
    if a.arch=="coarse_fine" and not a.init_checkpoint:
        p.error("V6 coarse_fine requires --init-checkpoint pointing to a trained V4 spatial model")
    torch.manual_seed(42);random.seed(42)
    device="cuda" if torch.cuda.is_available() else "cpu"
    train=DataLoader(ReferringDataset(a.train,a.image_root),batch_size=a.batch_size,
                     shuffle=True,collate_fn=collate,num_workers=0)
    val=DataLoader(ReferringDataset(a.val,a.image_root),batch_size=a.batch_size,
                   shuffle=False,collate_fn=collate,num_workers=0)
    model=CLIPGrounder(a.model,arch=a.arch).to(device)
    if a.init_checkpoint:
        ck=torch.load(a.init_checkpoint,map_location="cpu",weights_only=True)
        if a.arch!="coarse_fine" or ck.get("arch")!="spatial":
            p.error("--init-checkpoint currently supports V4 spatial -> V6 coarse_fine only")
        if ck["model_name"]!=a.model:
            p.error("V4 checkpoint model_name must match --model")
        missing,unexpected=model.load_state_dict(ck["head"],strict=False)
        allowed_missing={k for k in model.state_dict() if k.startswith("clip.") or k.startswith("refiner.")}
        if set(missing)!=allowed_missing or unexpected:
            raise ValueError(f"Incompatible V4 initialization: missing={missing}, unexpected={unexpected}")
        print(f"Initialized V6 coarse head from {a.init_checkpoint}; refiner starts at zero delta")
    if a.freeze_coarse:
        model.freeze_coarse()
    opt=torch.optim.AdamW((v for v in model.parameters() if v.requires_grad),lr=a.lr)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    best=(-1.,-1.)
    for epoch in range(1,a.epochs+1):
        model.train();losses=[]
        for batch in tqdm(train,desc=f"Epoch {epoch}"):
            opt.zero_grad(set_to_none=True)
            pred,aux=model(batch["images"],batch["texts"],return_aux=True)
            loss,_=grounding_loss(pred,batch["boxes"].to(device),aux,
                                   loss_type=a.loss,spatial_weight=a.spatial_weight)
            loss.backward();opt.step();losses.append(loss.item())
        metrics=evaluate(model,val,device)
        print(json.dumps({"epoch":epoch,"loss":sum(losses)/len(losses),**metrics}))
        current=(metrics["acc_at_0.5"],metrics["mean_iou"])
        if current>best:
            best=current
            head={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith("clip.")}
            torch.save({"head":head,"model_name":a.model,"arch":a.arch,"hidden":256,
                        "metrics":metrics,"epoch":epoch,"loss_type":a.loss,
                        "spatial_weight":a.spatial_weight,
                        "init_checkpoint":a.init_checkpoint,
                        "freeze_coarse":a.freeze_coarse},path)
    print("Best checkpoint saved:",path,"device:",device)

if __name__=="__main__":
    main()
