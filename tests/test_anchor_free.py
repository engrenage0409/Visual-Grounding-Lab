import torch
from torch import nn
from grounding.anchor_free import (grid_points, positive_point_mask,
                                   anchor_free_candidates, anchor_free_loss)


def test_anchor_free_candidates_and_gradients():
    torch.manual_seed(42)
    b,g,d=2,14,16
    feats=torch.randn(b,g*g,d,requires_grad=True)
    cls=nn.Linear(d,1)
    reg=nn.Linear(d,4)
    quality=nn.Linear(d,1)
    cls_logits,q_logits,boxes,scores=anchor_free_candidates(feats,cls,reg,quality,g)
    assert cls_logits.shape==(b,196)
    assert q_logits.shape==(b,196)
    assert boxes.shape==(b,196,4)
    assert scores.shape==(b,196)
    assert bool(((boxes>=0)&(boxes<=1)).all())
    assert bool((boxes[...,2:]>=boxes[...,:2]).all())
    target=torch.tensor([[.1,.1,.3,.3],[.6,.6,.85,.85]])
    indices=scores.argmax(dim=1)
    pred=boxes[torch.arange(b),indices]
    loss,ious=anchor_free_loss(pred,target,
        {"af_class_logits":cls_logits,"af_quality_logits":q_logits,"region_boxes":boxes},region_weight=1.)
    assert torch.isfinite(loss) and ious.shape==(b,)
    loss.backward()
    assert feats.grad is not None and feats.grad.abs().sum()>0
    assert cls.weight.grad is not None and cls.weight.grad.abs().sum()>0
    assert reg.weight.grad is not None and reg.weight.grad.abs().sum()>0
    assert quality.weight.grad is not None and quality.weight.grad.abs().sum()>0


def test_positive_sampling_and_small_target_fallback():
    target=torch.tensor([[.25,.25,.55,.55],[.125,.125,.126,.126]])
    positives=positive_point_mask(target,14)
    assert positives.shape==(2,196)
    assert positives[0].sum()>1
    assert positives[1].sum()>=1
    points=grid_points(14,target.device,target.dtype)
    chosen=points[positives[0]]
    assert bool(((chosen[:,0]>=.25)&(chosen[:,0]<=.55)).all())
    assert bool(((chosen[:,1]>=.25)&(chosen[:,1]<=.55)).all())
