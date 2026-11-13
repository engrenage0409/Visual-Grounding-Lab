"""V7.2: ground-truth-free V6 features -> generative or regression box head.

Two independently trained heads receive exactly the same frozen pretrained V6
visual/text-conditioned spatial tokens AND its predicted box. No GT box is
passed to the encoder. This is a diagnostic of coordinate serialization, not
a reproduction of Qwen-VL and not a test of grounding without location priors.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .checkpoints import load_grounder
from .generative import teacher_forcing_sequences, tokens_to_boxes
from .geometry import box_iou_aligned, generalized_iou_aligned
from .spatial import center_size_to_valid_xyxy
from .geometry import xyxy_to_cxcywh


class SpatialToBox(nn.Module):
    def __init__(self, v6_checkpoint, mode="generative", hidden=256,
                 coord_max=100, layers=2, heads=4, dropout=.1, device="cpu"):
        super().__init__()
        if mode not in ("generative","regression"):
            raise ValueError("mode must be generative or regression")
        if coord_max not in (100,224,1000):
            raise ValueError("coord_max must be 100, 224, or 1000")
        self.v6, ck = load_grounder(v6_checkpoint, device)
        if ck.get("arch")!="coarse_fine":
            raise ValueError("V7.2 requires a trained V6 coarse_fine checkpoint")
        self.v6.requires_grad_(False)
        self.v6.eval()
        self.mode=mode
        self.hidden=hidden
        self.coord_max=coord_max
        self.layers=layers
        self.heads=heads
        self.dropout=dropout
        self.bos_id=coord_max+1
        self.eos_id=coord_max+2
        self.vocab_size=coord_max+3
        source_dim=self.v6.text_proj.out_features
        self.spatial_proj=nn.Linear(source_dim,hidden)
        self.box_proj=nn.Linear(4,hidden)
        self.text_proj=nn.Linear(source_dim,hidden)
        self.memory_norm=nn.LayerNorm(hidden)
        if mode=="generative":
            self.token_embedding=nn.Embedding(self.vocab_size,hidden)
            self.position_embedding=nn.Embedding(5,hidden)
            block=nn.TransformerDecoderLayer(hidden,heads,hidden*4,
                   dropout=dropout,batch_first=True,norm_first=True)
            self.decoder=nn.TransformerDecoder(block,layers)
            self.output_norm=nn.LayerNorm(hidden)
            self.lm_head=nn.Linear(hidden,self.vocab_size)
        else:
            # Residual wrt the frozen V6 prediction. Zero starts exactly at V6.
            self.regression_head=nn.Sequential(
                nn.LayerNorm(hidden*3+4),nn.Linear(hidden*3+4,hidden),
                nn.GELU(),nn.Linear(hidden,4))
            nn.init.zeros_(self.regression_head[-1].weight)
            nn.init.zeros_(self.regression_head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.v6.eval()
        return self

    def encode(self, images, texts):
        with torch.no_grad():
            v6_box, aux = self.v6(images, texts, return_aux=True)
            tokens=aux["conditioned_tokens"]
            query=aux["text_query"]
            probs=aux["spatial_probs"]
        spatial=self.spatial_proj(tokens)
        text=self.text_proj(query)
        box=self.box_proj(v6_box)
        memory=self.memory_norm(torch.cat([spatial, text[:,None], box[:,None]],dim=1))
        weighted=(spatial*probs[...,None]).sum(dim=1)
        return memory, v6_box, weighted, text, box

    def decode(self, input_ids, memory):
        length=input_ids.shape[1]
        emb=self.token_embedding(input_ids)+self.position_embedding(
            torch.arange(length,device=input_ids.device))[None]
        mask=torch.triu(torch.ones(length,length,device=input_ids.device,
                                    dtype=torch.bool),diagonal=1)
        decoded=self.decoder(emb,memory,tgt_mask=mask)
        return self.lm_head(self.output_norm(decoded))

    def regress(self, base, weighted, text, box):
        delta=torch.tanh(self.regression_head(
            torch.cat([weighted,text,box,base],dim=-1)))
        geometry=xyxy_to_cxcywh(base)
        center=(geometry[:,:2]+.15*delta[:,:2]).clamp(0,1)
        size=(geometry[:,2:].clamp(min=1e-5)*torch.exp(.5*delta[:,2:])).clamp(max=1)
        return center_size_to_valid_xyxy(center,size)

    def forward(self, images, texts, targets):
        memory,base,weighted,text,box=self.encode(images,texts)
        if self.mode=="regression":
            predicted=self.regress(base,weighted,text,box)
            loss=F.l1_loss(predicted,targets)+(1-generalized_iou_aligned(predicted,targets)).mean()
            return {"loss":loss,"boxes":predicted,"v6_boxes":base}
        source,truth=teacher_forcing_sequences(targets,self.coord_max)
        logits=self.decode(source.to(memory.device),memory)
        loss=F.cross_entropy(logits.reshape(-1,self.vocab_size),truth.reshape(-1))
        argmax=logits[:,:4,:self.coord_max+1].argmax(-1)
        teacher_boxes=tokens_to_boxes(argmax,self.coord_max)
        return {"loss":loss,"teacher_boxes":teacher_boxes,
                "teacher_coordinate_accuracy":(argmax==truth[:,:4]).float().mean(),
                "teacher_eos_accuracy":(logits[:,4].argmax(-1)==self.eos_id).float().mean(),
                "v6_boxes":base}

    @torch.no_grad()
    def predict(self, images, texts, constrain_box=False):
        memory,base,weighted,text,box=self.encode(images,texts)
        if self.mode=="regression":
            return {"boxes":self.regress(base,weighted,text,box),
                    "v6_boxes":base}
        ids=torch.full((base.shape[0],1),self.bos_id,dtype=torch.long,device=base.device)
        selected=[]
        for step in range(5):
            scores=self.decode(ids,memory)[:,-1]
            if step<4:
                scores=scores[:,:self.coord_max+1]
                if constrain_box and step<2:
                    scores=scores.clone()
                    scores[:,self.coord_max]=float("-inf")
                if constrain_box and step>=2:
                    prev=selected[step-2]
                    index=torch.arange(self.coord_max+1,device=scores.device)
                    scores=scores.masked_fill(index[None]<=prev[:,None],float("-inf"))
            new=scores.argmax(-1)
            selected.append(new)
            if step<4:
                ids=torch.cat([ids,new[:,None]],dim=1)
        tokens=torch.stack(selected,dim=1)
        return {"boxes":tokens_to_boxes(tokens[:,:4],self.coord_max),
                "v6_boxes":base,"tokens":tokens,
                "eos_correct":tokens[:,4]==self.eos_id}
