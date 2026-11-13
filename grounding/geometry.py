import torch

def xyxy_to_cxcywh(box):
    x1,y1,x2,y2=box.unbind(dim=-1)
    return torch.stack(((x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1),dim=-1)

def cxcywh_to_xyxy(box):
    cx,cy,w,h=box.unbind(dim=-1)
    return torch.stack((cx-w/2,cy-h/2,cx+w/2,cy+h/2),dim=-1)

def box_iou_aligned(pred,target,eps=1e-7):
    lt=torch.maximum(pred[...,:2],target[...,:2])
    rb=torch.minimum(pred[...,2:],target[...,2:])
    wh=(rb-lt).clamp(min=0)
    inter=wh[...,0]*wh[...,1]
    pw=(pred[...,2:]-pred[...,:2]).clamp(min=0)
    tw=(target[...,2:]-target[...,:2]).clamp(min=0)
    union=pw[...,0]*pw[...,1]+tw[...,0]*tw[...,1]-inter
    return inter/union.clamp(min=eps)

def generalized_iou_aligned(pred,target,eps=1e-7):
    """GIoU provides a geometric gradient even for non-overlapping boxes."""
    iou=box_iou_aligned(pred,target,eps)
    lt=torch.minimum(pred[...,:2],target[...,:2])
    rb=torch.maximum(pred[...,2:],target[...,2:])
    wh=(rb-lt).clamp(min=0)
    enclosing=wh[...,0]*wh[...,1]
    pwh=(pred[...,2:]-pred[...,:2]).clamp(min=0)
    twh=(target[...,2:]-target[...,:2]).clamp(min=0)
    ilt=torch.maximum(pred[...,:2],target[...,:2])
    irb=torch.minimum(pred[...,2:],target[...,2:])
    iwh=(irb-ilt).clamp(min=0)
    inter=iwh[...,0]*iwh[...,1]
    union=pwh[...,0]*pwh[...,1]+twh[...,0]*twh[...,1]-inter
    return iou-(enclosing-union)/enclosing.clamp(min=eps)
