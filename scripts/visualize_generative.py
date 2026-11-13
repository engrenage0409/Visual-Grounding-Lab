"""Visualize V7 predicted coordinates and decoded box on one input image."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from grounding.generative_checkpoints import load_generative


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/v7_generative_b16.pt")
    p.add_argument("--image",required=True)
    p.add_argument("--text",required=True)
    p.add_argument("--output",default="results/v7_generative.png")
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,_=load_generative(a.checkpoint,device)
    with Image.open(a.image) as img:
        im=img.convert("RGB")
    with torch.inference_mode():
        result=model.generate([im],[a.text])
    coords=result["tokens"][0,:4].cpu().tolist()
    box=result["boxes"][0].cpu().tolist()
    width,height=im.size
    pixel=[box[0]*width,box[1]*height,box[2]*width,box[3]*height]
    valid=pixel[2]>pixel[0] and pixel[3]>pixel[1]
    canvas=im.copy()
    draw=ImageDraw.Draw(canvas)
    if valid:
        draw.rectangle(pixel,outline=(255,35,40),width=max(2,width//128))
    draw.text((5,5),a.text,fill=(255,35,40))
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(out)
    print(json.dumps({"image":a.image,"expression":a.text,
                      "coordinate_tokens":coords,"bbox_xyxy_normalized":box,
                      "bbox_xyxy_pixels":pixel,"valid_box":valid,
                      "eos_correct":bool(result["eos_correct"][0]),
                      "output":str(out)},indent=2))


if __name__=="__main__":
    main()
