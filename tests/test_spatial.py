import torch
from grounding.spatial import patch_target,attention_centers,spatial_cross_entropy
from grounding.geometry import generalized_iou_aligned
from grounding.engine import grounding_loss

def test_target_distribution_and_center():
    boxes=torch.tensor([[0.0,0.0,1.0,1.0],[0.0,0.0,1/7,1/7]])
    target=patch_target(boxes)
    assert target.shape==(2,49)
    assert torch.allclose(target.sum(-1),torch.ones(2))
    assert torch.allclose(target[1,0],torch.tensor(1.))
    center=attention_centers(target,7)
    assert torch.allclose(center[0],torch.tensor([.5,.5]),atol=1e-5)

def test_giou_non_overlapping_and_gradients():
    pred=torch.tensor([[0.0,0.0,0.2,0.2]],requires_grad=True)
    truth=torch.tensor([[.7,.7,.9,.9]])
    giou=generalized_iou_aligned(pred,truth)
    assert giou.item()<0
    loss=(1-giou).mean()
    loss.backward()
    assert pred.grad.abs().sum()>0

def test_spatial_loss_gradients():
    logits=torch.zeros(1,49,requires_grad=True)
    target=torch.tensor([[.2,.3,.5,.6]])
    loss=spatial_cross_entropy(logits,target)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum()>0
    boxes=torch.tensor([[.2,.3,.6,.7]],requires_grad=True)
    total,_=grounding_loss(boxes,target,{"spatial_logits":logits})
    assert torch.isfinite(total)
