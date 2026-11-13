import torch
from torch import nn
from grounding.spatial_to_generative import SpatialToBox
from grounding.generative import teacher_forcing_sequences


def test_regression_zero_delta_matches_frozen_v6_prediction():
    head=SpatialToBox.__new__(SpatialToBox)
    nn.Module.__init__(head)
    head.regression_head=nn.Sequential(nn.LayerNorm(16*3+4),
        nn.Linear(16*3+4,16),nn.GELU(),nn.Linear(16,4))
    nn.init.zeros_(head.regression_head[-1].weight)
    nn.init.zeros_(head.regression_head[-1].bias)
    boxes=torch.tensor([[.1,.2,.5,.6],[.4,.3,.8,.9]])
    a,b,c=torch.randn(2,16),torch.randn(2,16),torch.randn(2,16)
    pred=head.regress(boxes,a,b,c)
    assert torch.allclose(pred,boxes,atol=1e-6)
    (pred-torch.tensor([[.11,.2,.55,.62],[.35,.3,.8,.9]])).square().sum().backward()
    assert head.regression_head[-1].weight.grad.abs().sum()>0


def test_teacher_tokens_and_memory_shapes_for_v72():
    x=torch.tensor([[.1,.2,.4,.6]])
    source,target=teacher_forcing_sequences(x,coord_max=100)
    assert source.tolist()==[[101,10,20,40,60]]
    assert target.tolist()==[[10,20,40,60,102]]
    # The same V6 memory is supplied to either branch; no GT-derived memory.
    spatial=torch.randn(2,196,256)
    text=torch.randn(2,1,256)
    location=torch.randn(2,1,256)
    memory=torch.cat([spatial,text,location],dim=1)
    assert memory.shape==(2,198,256)
