"""Synthetic colored-rectangle data for smoke testing only."""
import argparse,json,random
from pathlib import Path
from PIL import Image,ImageDraw

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="data/toy")
    p.add_argument("--train",type=int,default=80)
    p.add_argument("--val",type=int,default=20)
    a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    rng=random.Random(42)
    palette={"red":(230,65,60),"blue":(55,105,225),"green":(35,170,80)}
    for split,n in (("train",a.train),("val",a.val)):
        with (root/f"{split}.jsonl").open("w",encoding="utf-8") as f:
            for i in range(n):
                im=Image.new("RGB",(224,224),(246,246,242))
                d=ImageDraw.Draw(im); boxes={}
                for j,color in enumerate(palette):
                    x=15+j*70+rng.randint(-5,5)
                    y=rng.randint(40,140)
                    boxes[color]=[x,y,x+45,y+50]
                    d.rectangle(boxes[color],fill=palette[color])
                selected=rng.choice(list(palette))
                name=f"{split}_{i:04d}.png"
                im.save(root/name)
                f.write(json.dumps({"image":name,"expression":f"the {selected} rectangle",
                                    "bbox_xyxy":boxes[selected]})+"\n")
    print("Generated",root)

if __name__=="__main__":
    main()
