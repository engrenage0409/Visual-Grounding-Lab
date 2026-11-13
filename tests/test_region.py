import torch
from torch import nn
from grounding.region import region_candidates, region_positive_indices, region_grounding_loss


def test_region_candidates_shape_and_topk():
    torch.manual_seed(42)
    b,grid,hidden=2,14,16
    features=torch.randn(b,grid*grid,hidden,requires_grad=True)
    score_head=nn.Linear(hidden,1)
    offset_head=nn.Linear(hidden,2)
    size_head=nn.Linear(hidden,2)
    logits,boxes=region_candidates(features,score_head,offset_head,size_head,grid)
    assert logits.shape==(b,196)
    assert boxes.shape==(b,196,4)
    assert ((boxes>=0)&(boxes<=1)).all()
    assert (boxes[...,2:]>=boxes[...,:2]).all()
    target=torch.tensor([[.1,.1,.25,.25],[.65,.65,.85,.85]])
    indices=region_positive_indices(target,grid)
    assert indices.shape==(b,)
    winner=logits.argmax(-1)
    predicted=boxes[torch.arange(b),winner]
    loss,ious=region_grounding_loss(predicted,target,
                                    {"region_logits":logits,"region_boxes":boxes},
                                    region_weight=1.0)
    assert loss.isfinite() and ious.shape==(b,)
    loss.backward()
    assert features.grad is not None
    assert score_head.weight.grad.abs().sum()>0
    assert offset_head.weight.grad.abs().sum()>0
    assert size_head.weight.grad.abs().sum()>0


def test_positive_grid_location():
    target=torch.tensor([[0.,0.,.1,.1],[.6,.6,.8,.8]])
    assert region_positive_indices(target,14).tolist()==[0,9*14+9]
