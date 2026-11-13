"""Train V7.2 frozen-V6 conditioned coordinate generation vs regression.

Keep V6 and previous V7 checkpoints unchanged. Each --mode is a separate run.
"""
import argparse
import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from grounding.data import ReferringDataset, collate
from grounding.geometry import box_iou_aligned
from grounding.spatial_to_generative import SpatialToBox


@torch.no_grad()
def evaluate(model, loader, device, constrain_box=False):
    model.eval()
    count=correct=base_correct=invalid=eos_count=0
    iou_sum=base_iou_sum=coord_mae=0.
    tf_correct=tf_iou_sum=0.
    for batch in loader:
        gt=batch["boxes"].to(device)
        result=model.predict(batch["images"],batch["texts"],constrain_box)
        box=result["boxes"]
        base=result["v6_boxes"]
        iou=box_iou_aligned(box,gt)
        base_iou=box_iou_aligned(base,gt)
        n=gt.shape[0]
        count+=n
        correct+=int((iou>=.5).sum().item())
        iou_sum+=float(iou.sum().item())
        base_correct+=int((base_iou>=.5).sum().item())
        base_iou_sum+=float(base_iou.sum().item())
        valid=(box[:,2]>box[:,0]) & (box[:,3]>box[:,1])
        invalid+=int((~valid).sum().item())
        coord_mae+=float((box-gt).abs().sum().item())
        if model.mode=="generative":
            eos_count+=int(result["eos_correct"].sum().item())
            tf=model(batch["images"],batch["texts"],gt)["teacher_boxes"]
            tf_iou=box_iou_aligned(tf,gt)
            tf_correct+=int((tf_iou>=.5).sum().item())
            tf_iou_sum+=float(tf_iou.sum().item())
    if count==0:
        raise ValueError("Empty validation set")
    report={"n":count,"mode":model.mode,"acc_at_0.5":correct/count,
            "mean_iou":iou_sum/count,
            "v6_base_acc_at_0.5":base_correct/count,
            "v6_base_mean_iou":base_iou_sum/count,
            "invalid_box_rate":invalid/count,
            "coordinate_mae_normalized":coord_mae/(4*count)}
    if model.mode=="generative":
        report.update({"eos_accuracy":eos_count/count,
                       "teacher_forced_acc_at_0.5":tf_correct/count,
                       "teacher_forced_mean_iou":tf_iou_sum/count,
                       "constrain_box":constrain_box})
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",required=True,choices=["generative","regression"])
    p.add_argument("--v6-checkpoint",default="checkpoints/v6_refine_b16.pt")
    p.add_argument("--train",default="data/v4_joint/train.jsonl")
    p.add_argument("--val",default="data/v4_mixed/val.jsonl")
    p.add_argument("--image-root",default="data/v4_mixed")
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--coord-max",type=int,choices=[100,224,1000],default=100)
    p.add_argument("--hidden",type=int,default=256)
    p.add_argument("--layers",type=int,default=2)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--dropout",type=float,default=.1)
    p.add_argument("--constrain-box",action="store_true")
    p.add_argument("--out",required=True)
    a=p.parse_args()
    torch.manual_seed(42);random.seed(42)
    device="cuda" if torch.cuda.is_available() else "cpu"
    train_loader=DataLoader(ReferringDataset(a.train,a.image_root),batch_size=a.batch_size,
                            shuffle=True,collate_fn=collate,num_workers=0)
    val_loader=DataLoader(ReferringDataset(a.val,a.image_root),batch_size=a.batch_size,
                          shuffle=False,collate_fn=collate,num_workers=0)
    model=SpatialToBox(a.v6_checkpoint,mode=a.mode,hidden=a.hidden,
                       coord_max=a.coord_max,layers=a.layers,heads=a.heads,
                       dropout=a.dropout,device=device).to(device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    best=(-1.,-1.)
    for epoch in range(1,a.epochs+1):
        model.train()
        loss_sum=0.;count=0
        for batch in tqdm(train_loader,desc=f"V7.2 {a.mode} epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            out=model(batch["images"],batch["texts"],batch["boxes"].to(device))
            out["loss"].backward()
            optimizer.step()
            loss_sum+=out["loss"].item();count+=1
        scores=evaluate(model,val_loader,device,constrain_box=a.constrain_box)
        print(json.dumps({"epoch":epoch,"train_loss":loss_sum/count,**scores}))
        current=(scores["acc_at_0.5"],scores["mean_iou"])
        if current>best:
            best=current
            head={k:v.detach().cpu() for k,v in model.state_dict().items()
                  if not k.startswith("v6.")}
            torch.save({"arch":"spatial_to_box","mode":a.mode,"coord_max":a.coord_max,
                        "hidden":a.hidden,"layers":a.layers,"heads":a.heads,
                        "dropout":a.dropout,"v6_checkpoint":a.v6_checkpoint,
                        "v6_model_name":model.v6.model_name,"head":head,
                        "epoch":epoch,"metrics":scores},path)
    print("Best checkpoint saved:",path)


if __name__=="__main__":
    main()
