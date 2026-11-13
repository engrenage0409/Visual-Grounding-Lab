import torch
from torch import nn
import torch.nn.functional as F
from transformers import CLIPModel,CLIPProcessor
from .geometry import cxcywh_to_xyxy
from .spatial import attention_centers,center_size_to_valid_xyxy
from .region import region_candidates
from .anchor_free import anchor_free_candidates
from .refinement import CoarseFineRefiner

class CLIPGrounder(nn.Module):
    """Frozen CLIP patch/text representations + trainable box head.

    arch=baseline retains the original single text-query cross-attention model.
    arch=spatial adds text-conditioned per-patch scores and spatial expectation.
    arch=region (V5) scores separate patch-anchored candidate boxes, avoiding
    averaging the centers of competing objects.
    """
    def __init__(self,model_name="openai/clip-vit-base-patch32",hidden=256,arch="baseline"):
        super().__init__()
        if arch not in ("baseline","spatial","region","region_af","coarse_fine"): raise ValueError("Unknown architecture")
        self.arch=arch
        self.model_name=model_name
        self.processor=CLIPProcessor.from_pretrained(model_name)
        self.clip=CLIPModel.from_pretrained(model_name)
        self.clip.requires_grad_(False)
        self.clip.eval()
        d=self.clip.config.projection_dim
        self.patch_proj=nn.Linear(d,hidden)
        self.text_proj=nn.Linear(d,hidden)
        if arch=="baseline":
            self.attn=nn.MultiheadAttention(hidden,num_heads=4,batch_first=True)
            self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),
                                    nn.GELU(),nn.Linear(hidden,4))
        else:
            vision_config=self.clip.config.vision_config
            image_size=vision_config.image_size
            patch_size=vision_config.patch_size
            if not isinstance(image_size,int) or not isinstance(patch_size,int) or image_size%patch_size:
                raise ValueError(f"Unsupported CLIP vision grid: image_size={image_size}, patch_size={patch_size}")
            self.grid=image_size//patch_size
            self.num_patches=self.grid**2
            self.patch_position=nn.Parameter(torch.randn(1,self.num_patches,hidden)*0.02)
            self.patch_norm=nn.LayerNorm(hidden)
            self.text_norm=nn.LayerNorm(hidden)
            self.patch_fusion=nn.Sequential(nn.Linear(hidden*2,hidden),nn.GELU(),
                                            nn.LayerNorm(hidden))
            self.score_head=nn.Linear(hidden,1)
            if arch in ("spatial","coarse_fine"):
                self.size_head=nn.Sequential(nn.Linear(hidden*2,hidden),nn.GELU(),
                                             nn.Linear(hidden,2))
                if arch=="coarse_fine":
                    self.refiner=CoarseFineRefiner(hidden)
            elif arch=="region":
                self.region_offset_head=nn.Linear(hidden,2)
                self.region_size_head=nn.Linear(hidden,2)
            else:
                self.af_distance_head=nn.Linear(hidden,4)
                self.af_quality_head=nn.Linear(hidden,1)
    def freeze_coarse(self):
        """V6: train only the refinement head while retaining the V4 prediction."""
        if self.arch!="coarse_fine":
            raise ValueError("freeze_coarse requires arch=coarse_fine")
        for name,param in self.named_parameters():
            param.requires_grad_(name.startswith("refiner."))

    def train(self,mode=True):
        super().train(mode)
        self.clip.eval()
        return self
    def forward(self,images,texts,return_aux=False):
        inputs=self.processor(images=images,text=texts,padding=True,truncation=True,
                              return_tensors="pt")
        device=next(self.parameters()).device
        inputs={k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            patch=self.clip.vision_model(pixel_values=inputs["pixel_values"]).last_hidden_state[:,1:,:]
            patch=self.clip.visual_projection(patch)
            text=self.clip.text_model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"]).pooler_output
            text=self.clip.text_projection(text)
        q=self.text_proj(text)
        kv=self.patch_proj(patch)
        if self.arch=="baseline":
            out,_=self.attn(q.unsqueeze(1),kv,kv,need_weights=False)
            boxes=cxcywh_to_xyxy(torch.sigmoid(self.head(out[:,0]))).clamp(0,1)
            return (boxes,{}) if return_aux else boxes
        if kv.shape[1]!=self.num_patches:
            raise ValueError(f"Expected {self.num_patches} patches ({self.grid}x{self.grid}), got {kv.shape[1]}")
        visual=self.patch_norm(kv+self.patch_position)
        query=self.text_norm(q)
        # Language features are included *per spatial token*, not only after pooling.
        conditioned=self.patch_fusion(torch.cat(
            [visual,query.unsqueeze(1).expand_as(visual)],dim=-1))
        logits=self.score_head(conditioned).squeeze(-1)
        probs=F.softmax(logits,dim=-1)
        if self.arch=="region_af":
            classification,quality,candidates,scores=anchor_free_candidates(
                conditioned,self.score_head,self.af_distance_head,self.af_quality_head,self.grid)
            topk=torch.topk(scores,k=min(5,self.num_patches),dim=-1)
            winner=topk.indices[:,0]
            batch_idx=torch.arange(scores.shape[0],device=scores.device)
            boxes=candidates[batch_idx,winner]
            aux={"af_class_logits":classification,"af_quality_logits":quality,
                 "region_boxes":candidates,"region_scores":scores,
                 "spatial_probs":scores / scores.sum(-1,keepdim=True).clamp(min=1e-8),
                 "topk_indices":topk.indices,"topk_scores":topk.values}
            return (boxes,aux) if return_aux else boxes
        if self.arch=="region":
            logits,candidates=region_candidates(conditioned,self.score_head,
                self.region_offset_head,self.region_size_head,self.grid)
            topk=torch.topk(logits,k=min(5,self.num_patches),dim=-1)
            winner=topk.indices[:,0]
            batch_idx=torch.arange(logits.shape[0],device=logits.device)
            boxes=candidates[batch_idx,winner]
            aux={"region_logits":logits,"region_boxes":candidates,
                 "spatial_probs":F.softmax(logits,dim=-1),
                 "topk_indices":topk.indices,"topk_scores":topk.values}
            return (boxes,aux) if return_aux else boxes
        center=attention_centers(probs,grid=self.grid)
        attended=(probs.unsqueeze(-1)*visual).sum(1)
        size=torch.sigmoid(self.size_head(torch.cat([attended,query],dim=-1)))
        boxes=center_size_to_valid_xyxy(center,size)
        aux={"spatial_logits":logits,"spatial_probs":probs}
        if self.arch=="coarse_fine":
            # Stop gradients to coarse box; independently trainable refinement.
            coarse_boxes=boxes
            refined,delta=self.refiner(visual,query,coarse_boxes.detach(),self.grid)
            aux={"coarse_boxes":coarse_boxes,"refine_delta":delta,
                 "spatial_probs":probs}
            boxes=refined
        return (boxes,aux) if return_aux else boxes
