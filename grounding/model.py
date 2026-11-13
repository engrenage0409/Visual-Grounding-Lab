import torch
from torch import nn
from transformers import CLIPModel,CLIPProcessor
from .geometry import cxcywh_to_xyxy

class CLIPGrounder(nn.Module):
    """Frozen CLIP patch/text embeddings + trainable cross-attention box head."""
    def __init__(self,model_name="openai/clip-vit-base-patch32",hidden=256):
        super().__init__()
        self.processor=CLIPProcessor.from_pretrained(model_name)
        self.clip=CLIPModel.from_pretrained(model_name)
        self.clip.requires_grad_(False)
        self.clip.eval()
        d=self.clip.config.projection_dim
        self.patch_proj=nn.Linear(d,hidden)
        self.text_proj=nn.Linear(d,hidden)
        self.attn=nn.MultiheadAttention(hidden,num_heads=4,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),
                                nn.GELU(),nn.Linear(hidden,4))
    def train(self,mode=True):
        super().train(mode)
        self.clip.eval()
        return self
    def forward(self,images,texts):
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
        q=self.text_proj(text).unsqueeze(1)
        kv=self.patch_proj(patch)
        out,_=self.attn(q,kv,kv,need_weights=False)
        return cxcywh_to_xyxy(torch.sigmoid(self.head(out[:,0]))).clamp(0,1)
