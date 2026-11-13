"""Convert existing JSONL into Qwen3-VL instruction-SFT examples."""
import argparse,json
from pathlib import Path
from PIL import Image

def convert(source,image_root,out):
    root=Path(image_root)
    target=Path(out);target.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with Path(source).open(encoding="utf-8") as inp,target.open("w",encoding="utf-8") as dst:
        for line in inp:
            if not line.strip():continue
            row=json.loads(line)
            file=Path(row["image"])
            if not file.is_absolute():file=root/file
            file=file.resolve(strict=True)
            with Image.open(file) as img:w,h=img.size
            b=row["bbox_xyxy"]
            if len(b)!=4 or not(0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):raise ValueError(file)
            coords=[round(v*1000/d) for v,d in zip(b,[w,h,w,h])]
            text=f"Locate {row['expression']} in the image. Return only JSON with bbox_2d as [x1,y1,x2,y2] in normalized 0-1000 coordinates."
            result={"messages":[{"role":"user","content":"<image>"+text},
                    {"role":"assistant","content":json.dumps({"bbox_2d":coords},separators=(",",":"))}],
                    "images":[str(file)]}
            dst.write(json.dumps(result,ensure_ascii=False)+"\n");n+=1
    print(f"Converted {n} items to {target}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True);p.add_argument("--image-root",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();convert(a.source,a.image_root,a.out)
