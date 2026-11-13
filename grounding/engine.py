import torch
import torch.nn.functional as F
from .geometry import box_iou_aligned

def grounding_loss(pred,target):
    iou=box_iou_aligned(pred,target)
    return F.l1_loss(pred,target)+(1-iou).mean(),iou

@torch.no_grad()
def evaluate(model,loader,device,threshold=0.5):
    model.eval()
    n=0; correct=0; total_iou=0.
    for batch in loader:
        pred=model(batch["images"],batch["texts"])
        iou=box_iou_aligned(pred,batch["boxes"].to(device))
        n+=len(iou)
        correct+=(iou>=threshold).sum().item()
        total_iou+=iou.sum().item()
    if not n: raise ValueError("No validation examples")
    return {"acc_at_0.5":correct/n,"mean_iou":total_iou/n,"n":n}
