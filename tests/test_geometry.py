import torch
from grounding.geometry import xyxy_to_cxcywh,cxcywh_to_xyxy,box_iou_aligned

def test_roundtrip():
    x=torch.tensor([[0.1,0.2,0.6,0.9]])
    assert torch.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(x)),x)

def test_iou():
    a=torch.tensor([[0.,0.,1.,1.],[0.,0.,0.5,0.5]])
    b=torch.tensor([[0.,0.,1.,1.],[0.5,0.5,1.,1.]])
    assert torch.allclose(box_iou_aligned(a,b),torch.tensor([1.,0.]))
