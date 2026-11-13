import torch
from grounding.refinement import CoarseFineRefiner, multiscale_roi_features, roi_grid


def test_multiscale_roi_shapes_and_gradients():
    torch.manual_seed(4)
    feats=torch.randn(2,196,16,requires_grad=True)
    boxes=torch.tensor([[.1,.2,.5,.6],[.5,.3,.9,.8]])
    pooled=multiscale_roi_features(feats,boxes,grid=14)
    assert pooled.shape==(2,32)
    pooled.sum().backward()
    assert feats.grad is not None and feats.grad.abs().sum()>0


def test_zero_init_matches_v4_and_refiner_learns():
    torch.manual_seed(4)
    head=CoarseFineRefiner(hidden=16)
    tokens=torch.randn(2,196,16)
    text=torch.randn(2,16)
    coarse=torch.tensor([[.1,.2,.5,.6],[.5,.3,.9,.8]])
    prediction,delta=head(tokens,text,coarse,grid=14)
    assert torch.allclose(prediction,coarse,atol=1e-6)
    assert torch.count_nonzero(delta)==0
    target=torch.tensor([[.12,.2,.52,.6],[.5,.32,.9,.82]])
    (prediction-target).square().sum().backward()
    assert head.refine_head[-1].weight.grad.abs().sum()>0


def test_roi_grid_normalized():
    g=roi_grid(torch.tensor([[0.,0.,1.,1.]]),samples=3)
    assert g.shape==(1,3,3,2)
    assert torch.allclose(g[0,0,0],torch.tensor([-1.,-1.]))
    assert torch.allclose(g[0,-1,-1],torch.tensor([1.,1.]))
