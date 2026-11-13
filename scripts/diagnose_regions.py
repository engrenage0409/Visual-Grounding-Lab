"""Diagnose region/region_af proposal quality vs ranking without oracle inference.

Oracle@K is evaluation-only: choose best-IoU candidate among top-K scores.
It uses ground truth and must never be reported as deployable inference accuracy.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from grounding.checkpoints import load_grounder
from grounding.data import ReferringDataset, collate
from grounding.geometry import box_iou_aligned


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--image-root",required=True)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--out",default=None)
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,ck=load_grounder(a.checkpoint,device)
    if ck.get("arch") not in ("region","region_af"):
        raise ValueError("Oracle Top-K requires region or region_af checkpoint")
    loader=DataLoader(ReferringDataset(a.data,a.image_root),batch_size=a.batch_size,
                      shuffle=False,collate_fn=collate,num_workers=0)
    ks=(1,3,5,10)
    totals={k:{"correct":0,"iou_sum":0.} for k in ks}
    count=0
    examples=[]
    with torch.inference_mode():
        for batch in loader:
            predicted,aux=model(batch["images"],batch["texts"],return_aux=True)
            candidates=aux["region_boxes"]
            scores=aux.get("region_scores")
            if scores is None:
                scores=aux["region_logits"]
            n,bins=candidates.shape[:2]
            truth=batch["boxes"].to(device)
            all_ious=box_iou_aligned(candidates.reshape(-1,4),
                                    truth[:,None,:].expand_as(candidates).reshape(-1,4)).reshape(n,bins)
            ranks=scores.argsort(dim=-1,descending=True)
            rows=torch.arange(n,device=device)[:,None]
            ranked_ious=all_ious[rows,ranks]
            actual=box_iou_aligned(predicted,truth)
            if not torch.allclose(actual,ranked_ious[:,0],atol=1e-5):
                raise AssertionError("top-1 candidate != model output")
            for k in ks:
                oracle=ranked_ious[:,:min(k,bins)].max(-1).values
                totals[k]["correct"]+=int((oracle>=.5).sum().item())
                totals[k]["iou_sum"]+=float(oracle.sum().item())
            for j in range(n):
                examples.append({"image":batch["paths"][j],"text":batch["texts"][j],
                                 "top1_iou":round(actual[j].item(),4),
                                 "oracle_top5_iou":round(ranked_ious[j,:min(5,bins)].max().item(),4),
                                 "oracle_all_iou":round(all_ious[j].max().item(),4)})
            count+=n
    report={"checkpoint":a.checkpoint,"arch":ck.get("arch"),"n":count,
            "oracle_topk":{str(k):{"acc_at_0.5":round(v["correct"]/count,4),
                                 "mean_iou":round(v["iou_sum"]/count,4)}
                            for k,v in totals.items()},
            "oracle_all":{"acc_at_0.5":round(sum(e["oracle_all_iou"]>=.5 for e in examples)/count,4),
                          "mean_iou":round(sum(e["oracle_all_iou"] for e in examples)/count,4)},
            "biggest_ranking_gaps":sorted(examples,key=lambda e:e["oracle_top5_iou"]-e["top1_iou"],reverse=True)[:10]}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if a.out:
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
