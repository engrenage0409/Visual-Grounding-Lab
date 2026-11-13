"""Reproduce the paired Qwen3-VL before/after summary from two evaluation CSVs.

Standard-library only; no pandas dependency. Can be rerun on new experiments.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        rows=list(csv.DictReader(stream))
    lookup={}
    for row in rows:
        key=(row["image"].replace("\\","/"), row["expression"])
        if key in lookup:
            raise ValueError(f"Duplicate sample in {path}: {key}")
        lookup[key]=row
    return lookup


def compare(before_path,after_path):
    before=load_csv(before_path)
    after=load_csv(after_path)
    if not before or before.keys()!=after.keys():
        raise ValueError("Before and after must contain the same non-empty (image, expression) keys")
    n=len(before)
    changes=defaultdict(int)
    shapes=defaultdict(lambda:{"n":0,"before_correct":0,"after_correct":0})
    failures=[]
    greater=lower=equal=0
    iou_before=iou_after=0.
    for key,old in before.items():
        new=after[key]
        if json.loads(old["gt_xyxy"])!=json.loads(new["gt_xyxy"]):
            raise ValueError(f"Mismatched GT for {key}")
        b=float(old["iou"]);a=float(new["iou"])
        old_ok=b>=.5;new_ok=a>=.5
        changes[f"{int(old_ok)}->{int(new_ok)}"]+=1
        if a>b+1e-9:greater+=1
        elif a<b-1e-9:lower+=1
        else:equal+=1
        iou_before+=b;iou_after+=a
        shape=next((x for x in ("circle","rectangle","triangle") if x in key[1].split()),"other")
        stat=shapes[shape]
        stat["n"]+=1
        stat["before_correct"]+=int(old_ok)
        stat["after_correct"]+=int(new_ok)
        if not new_ok:
            failures.append({"image":key[0],"expression":key[1],
                            "iou_before":round(b,6),"iou_after":round(a,6),
                            "correct_before":old_ok})
    return {
        "n":n,
        "acc_before":(changes["1->1"]+changes["1->0"])/n,
        "acc_after":(changes["1->1"]+changes["0->1"])/n,
        "mean_iou_before":iou_before/n,
        "mean_iou_after":iou_after/n,
        "transitions":dict(sorted(changes.items())),
        "iou_improved":greater,"iou_regressed":lower,"iou_unchanged":equal,
        "by_shape":dict(sorted(shapes.items())),"failures_after":failures,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--before",default="results/qwen_before.csv")
    parser.add_argument("--after",default="results/qwen_after.csv")
    parser.add_argument("--out",default="results/qwen_paired_summary.json")
    a=parser.parse_args()
    report=compare(a.before,a.after)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
