import math
import argparse,json
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from grounding.checkpoints import load_grounder

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--image",required=True)
    p.add_argument("--text",required=True)
    p.add_argument("--checkpoint",default="checkpoints/grounder.pt")
    p.add_argument("--output",default="results/grounding_preview.png")
    p.add_argument("--heatmap",default=None,help="Optional spatial heatmap overlay PNG (auto-detects patch grid)")
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,ck=load_grounder(a.checkpoint,device)
    with Image.open(a.image) as f: im=f.convert("RGB")
    with torch.inference_mode():
        boxes,aux=model([im],[a.text],return_aux=True)
        box=boxes[0].cpu().tolist()
    w,h=im.size
    xyxy=[box[0]*w,box[1]*h,box[2]*w,box[3]*h]
    canvas=im.copy();draw=ImageDraw.Draw(canvas)
    draw.rectangle(xyxy,outline=(250,35,45),width=max(2,w//128))
    draw.text((5,5),a.text,fill=(250,35,45))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(out)
    result={"output":str(out),"arch":ck.get("arch","baseline"),
            "bbox_xyxy_normalized":box,"bbox_xyxy_pixels":xyxy}
    if a.heatmap and "spatial_probs" in aux:
        import numpy as np
        patch_probs=aux["spatial_probs"][0]
        grid=math.isqrt(patch_probs.numel())
        if grid*grid!=patch_probs.numel():
            raise ValueError(f"Spatial heatmap requires a square grid, got {patch_probs.numel()} patches")
        prob=patch_probs.reshape(grid,grid).cpu().numpy()
        prob=(prob-prob.min())/(np.ptp(prob)+1e-9)
        heat=Image.fromarray(np.uint8(prob*255),"L").resize(im.size,Image.Resampling.BILINEAR)
        red=Image.new("RGB",im.size,(255,30,30))
        overlay=Image.composite(red,im,heat.point(lambda x:int(x*0.58)))
        heat_path=Path(a.heatmap);heat_path.parent.mkdir(parents=True,exist_ok=True)
        overlay.save(heat_path)
        result["heatmap"]=str(heat_path)
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
