"""Spatial supervision for a 7x7 CLIP patch grid and valid box geometry."""
import torch
import torch.nn.functional as F

def patch_target(boxes,grid=7):
    """Soft target: intersection area of GT box with each normalized patch cell.
    Shape [B,grid*grid], sums to 1. Works for targets covering multiple patches.
    """
    if boxes.ndim!=2 or boxes.shape[1]!=4: raise ValueError("boxes must be [B,4] xyxy")
    device,dtype=boxes.device,boxes.dtype
    y,x=torch.meshgrid(torch.arange(grid,device=device,dtype=dtype),
                       torch.arange(grid,device=device,dtype=dtype),indexing="ij")
    x0=(x/grid).reshape(1,-1);x1=((x+1)/grid).reshape(1,-1)
    y0=(y/grid).reshape(1,-1);y1=((y+1)/grid).reshape(1,-1)
    left=torch.maximum(boxes[:,0:1],x0)
    top=torch.maximum(boxes[:,1:2],y0)
    right=torch.minimum(boxes[:,2:3],x1)
    bottom=torch.minimum(boxes[:,3:4],y1)
    area=(right-left).clamp(min=0)*(bottom-top).clamp(min=0)
    return area/area.sum(-1,keepdim=True).clamp(min=1e-8)

def attention_centers(probabilities,grid):
    """Differentiable expectation of patch center coordinates."""
    coords=(torch.arange(grid,device=probabilities.device,dtype=probabilities.dtype)+0.5)/grid
    y,x=torch.meshgrid(coords,coords,indexing="ij")
    return torch.stack([(probabilities*x.flatten()).sum(-1),
                        (probabilities*y.flatten()).sum(-1)],dim=-1)

def center_size_to_valid_xyxy(center,size):
    """Center and size in (0,1) -> valid xyxy, clipped to image by construction."""
    half=size/2
    xy1=(center-half).clamp(0,1)
    xy2=(center+half).clamp(0,1)
    return torch.cat([xy1,xy2],dim=-1)

def spatial_cross_entropy(logits,boxes,grid=7):
    target=patch_target(boxes,grid)
    return -(target*F.log_softmax(logits,dim=-1)).sum(-1).mean()
