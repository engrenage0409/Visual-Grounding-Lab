import torch
import torch.nn.functional as F
from .geometry import box_iou_aligned,generalized_iou_aligned
from .spatial import spatial_cross_entropy

def grounding_loss(pred,target,aux=None,loss_type="giou",spatial_weight=1.0):
    """One-box regression. Optional spatial CE target is area-overlap soft patch mask."""
    iou=box_iou_aligned(pred,target)
    geometric=(1-generalized_iou_aligned(pred,target)).mean() if loss_type=="giou" else (1-iou).mean()
    box_l1=F.l1_loss(pred,target)
    loss=box_l1+geometric
    if aux and "spatial_logits" in aux:
        loss=loss+spatial_weight*spatial_cross_entropy(aux["spatial_logits"],target,grid=7)
    return loss,iou

@torch.no_grad()
def evaluate(model,loader,device,threshold=0.5):
    model.eval()
    n=0;correct=0;total_iou=0.
    for batch in loader:
        pred=model(batch["images"],batch["texts"])
        iou=box_iou_aligned(pred,batch["boxes"].to(device))
        n+=len(iou);correct+=(iou>=threshold).sum().item()
        total_iou+=iou.sum().item()
    if not n: raise ValueError("No validation examples")
    return {"acc_at_0.5":correct/n,"mean_iou":total_iou/n,"n":n}
