import argparse,json
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from grounding.model import CLIPGrounder

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--image",required=True)
    p.add_argument("--text",required=True)
    p.add_argument("--checkpoint",default="checkpoints/grounder.pt")
    p.add_argument("--output",default="results/grounding_preview.png")
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    model=CLIPGrounder(ck["model_name"]).to(device)
    model.load_state_dict(ck["head"],strict=False);model.eval()
    with Image.open(a.image) as f: im=f.convert("RGB")
    with torch.inference_mode():
        box=model([im],[a.text])[0].cpu().tolist()
    w,h=im.size
    xyxy=[box[0]*w,box[1]*h,box[2]*w,box[3]*h]
    canvas=im.copy();draw=ImageDraw.Draw(canvas)
    draw.rectangle(xyxy,outline=(250,35,45),width=max(2,w//128))
    draw.text((5,5),a.text,fill=(250,35,45))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(out)
    print(json.dumps({"output":str(out),"bbox_xyxy_pixels":xyxy}))

if __name__=="__main__":
    main()
