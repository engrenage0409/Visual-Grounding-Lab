import torch
from torch import nn
from grounding.generative import (
    BOS, EOS, VOCAB_SIZE, OUTPUT_LENGTH, quantize_boxes,
    teacher_forcing_sequences, tokens_to_boxes, GenerativeGrounder,
)


def test_coordinate_token_roundtrip_and_teacher_forcing():
    boxes=torch.tensor([[0.,.125,.75,1.],[.11,.22,.33,.44]])
    source,target=teacher_forcing_sequences(boxes)
    assert source.shape==(2,5) and target.shape==(2,5)
    assert (source[:,0]==BOS).all()
    assert (target[:,-1]==EOS).all()
    assert torch.equal(source[:,1:],target[:,:4])
    recovered=tokens_to_boxes(target[:,:4])
    assert torch.allclose(recovered,boxes,atol=.0005)


def test_decoder_causal_mask_and_ce_gradient_without_clip_download():
    # Build decoder-only components without calling pretrained CLIP.
    torch.manual_seed(17)
    fake=GenerativeGrounder.__new__(GenerativeGrounder)
    nn.Module.__init__(fake)
    fake.token_embedding=nn.Embedding(VOCAB_SIZE,16)
    fake.position_embedding=nn.Embedding(OUTPUT_LENGTH,16)
    layer=nn.TransformerDecoderLayer(d_model=16,nhead=4,dim_feedforward=64,
                                      batch_first=True,dropout=0.)
    fake.decoder=nn.TransformerDecoder(layer,num_layers=1)
    fake.output_norm=nn.LayerNorm(16)
    fake.lm_head=nn.Linear(16,VOCAB_SIZE)
    memory=torch.randn(2,4,16)
    inputs,targets=teacher_forcing_sequences(torch.tensor([[.1,.2,.3,.4],[.2,.3,.5,.6]]))
    fake.eval()
    a=fake.decode(inputs,memory)
    changed=inputs.clone()
    changed[:,-1]=999
    b=fake.decode(changed,memory)
    # Earlier predictions cannot see a later changed token.
    assert torch.allclose(a[:,:-1],b[:,:-1],atol=1e-5)
    loss=nn.functional.cross_entropy(a.reshape(-1,VOCAB_SIZE),targets.reshape(-1))
    loss.backward()
    assert fake.lm_head.weight.grad.abs().sum()>0


def test_v71_coordinate_vocab_ablation():
    boxes=torch.tensor([[0.121, 0.233, 0.711, 0.944]])
    for resolution in (100,224,1000):
        x,y=teacher_forcing_sequences(boxes,coord_max=resolution)
        assert x[0,0]==resolution+1
        assert y[0,-1]==resolution+2
        assert y.max()<=resolution+2
        assert torch.equal(x[:,1:],y[:,:4])
        reconstructed=tokens_to_boxes(y[:,:4],coord_max=resolution)
        assert (reconstructed-boxes).abs().max() <= .5/resolution+1e-6


def test_v71_box_coordinate_metrics_from_logits():
    # A correct EOS says nothing about the four predicted coordinates.
    resolution=100
    gt=torch.tensor([[.1,.2,.7,.8]])
    _,targets=teacher_forcing_sequences(gt,coord_max=resolution)
    prediction=targets.clone()
    prediction[:,:4]=0
    full=(prediction==targets).float().mean().item()
    coordinate=(prediction[:,:4]==targets[:,:4]).float().mean().item()
    eos=(prediction[:,4]==targets[:,4]).float().mean().item()
    assert abs(full-.2)<1e-6
    assert coordinate==0.
    assert eos==1.
