"""Evaluate Qwen3-VL before or after Grounding LoRA on the same JSONL.

This uses 4-bit quantized 2B Instruct (8GB-class GPU), optional PEFT adapter,
the exact same prompt/coordinate convention as prepare_qwen_grounding.py,
and logs parse failures instead of treating unparsable answers as valid boxes.
"""
import argparse,csv,json,re
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from grounding.geometry import box_iou_aligned


def parse_box(answer):
    # Accept an object or single-item object list, but not arbitrary numbers
    # appearing in explanations, to avoid scoring unsupported hallucinated text.
    try:
        match=re.search(r'\{[^{}]*"bbox_2d"\s*:\s*\[[^\]]+\][^{}]*\}',answer,re.S)
        if not match:return None
        value=json.loads(match.group(0))["bbox_2d"]
        if not isinstance(value,list) or len(value)!=4:return None
        if any(type(x) not in (int,float) or not(0<=x<=1000) for x in value):return None
        return [float(v)/1000 for v in value]
    except (ValueError,TypeError,KeyError,json.JSONDecodeError):
        return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/v4_hard/val.jsonl")
    p.add_argument("--image-root",default="data/v4_hard")
    p.add_argument("--model",default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--adapter",default=None,help="Optional ms-swift LoRA adapter checkpoint")
    p.add_argument("--limit",type=int,default=100)
    p.add_argument("--max-new-tokens",type=int,default=100)
    p.add_argument("--out",default="results/qwen_grounding_baseline.csv")
    a=p.parse_args()
    from peft import PeftModel
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForImageTextToText.from_pretrained(
        a.model,quantization_config=q,device_map="auto",dtype=torch.bfloat16)
    if a.adapter:model=PeftModel.from_pretrained(model,a.adapter)
    model.eval()
    processor=AutoProcessor.from_pretrained(a.model)
    rows=[json.loads(line) for line in Path(a.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.limit>0:rows=rows[:a.limit]
    dst=Path(a.out);dst.parent.mkdir(parents=True,exist_ok=True)
    results=[]
    for i,row in enumerate(rows):
        file=Path(row["image"])
        if not file.is_absolute():file=Path(a.image_root)/file
        with Image.open(file) as opened:
            img=opened.convert("RGB")
        w,h=img.size
        gt=[float(x)/scale for x,scale in zip(row["bbox_xyxy"],[w,h,w,h])]
        prompt=f"Locate {row['expression']} in the image. Return only JSON with bbox_2d as [x1,y1,x2,y2] in normalized 0-1000 coordinates."
        messages=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt}]}]
        inputs=processor.apply_chat_template(messages,tokenize=True,
                        add_generation_prompt=True,return_dict=True,return_tensors="pt")
        inputs=inputs.to(model.device)
        with torch.inference_mode():
            ids=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False)
        decoded=processor.batch_decode(ids[:,inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        box=parse_box(decoded)
        valid=box is not None and box[2]>box[0] and box[3]>box[1]
        iou=float(box_iou_aligned(torch.tensor([box]),torch.tensor([gt]))[0]) if valid else 0.
        results.append({"image":str(file),"expression":row["expression"],
                       "gt_xyxy":json.dumps(gt),"pred_xyxy":json.dumps(box),
                       "raw_output":decoded,"parse_ok":box is not None,
                       "valid_box":valid,"iou":iou,"correct":iou>=.5})
        if (i+1)%10==0:print(f"{i+1}/{len(rows)} complete",flush=True)
    with dst.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]) if results else [])
        writer.writeheader();writer.writerows(results)
    n=len(results)
    if not n:raise ValueError("Empty evaluation data")
    summary={"n":n,"acc_at_0.5":sum(r["correct"] for r in results)/n,
             "mean_iou":sum(r["iou"] for r in results)/n,
             "parse_success":sum(r["parse_ok"] for r in results)/n,
             "valid_box_rate":sum(r["valid_box"] for r in results)/n,
             "model":a.model,"adapter":a.adapter,"out":str(dst)}
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
