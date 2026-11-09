#!/usr/bin/env python3
"""Evaluate SnakeWatch detector score thresholds on a YOLO-format test/validation set.

Expected layout:
  images/val/**/*.jpg
  labels/val/**/*.txt
Image and label relative paths must match (with .txt replacing image extension).
YOLO label rows: class_id x_center y_center width height (normalized 0..1).
For this single-class snake detector, class_id 0 is treated as snake; empty/missing
label files mean a negative image. Use a held-out test set only for final reporting.

Optional --groups-csv: CSV with columns image,group. `image` can be the relative image
path (relative to --images) or basename. `group` can be e.g. rope, hose, snake, background.
This supports per-hard-negative summaries when the dataset has such metadata.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_yolo_labels(path: Path, width: int, height: int):
    boxes=[]
    if not path.exists(): return boxes
    for line in path.read_text().splitlines():
        bits=line.strip().split()
        if len(bits) < 5: continue
        cls=int(float(bits[0]))
        if cls != 0: continue
        xc,yc,bw,bh=map(float,bits[1:5])
        x1=(xc-bw/2)*width; y1=(yc-bh/2)*height
        x2=(xc+bw/2)*width; y2=(yc+bh/2)*height
        boxes.append([max(0,x1),max(0,y1),min(width,x2),min(height,y2)])
    return boxes

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(aa+bb-inter,1e-9)

def greedy_match(preds, gt, iou_threshold):
    pairs=[]
    for pi,p in enumerate(preds):
        for gi,g in enumerate(gt):
            v=iou(p,g)
            if v >= iou_threshold: pairs.append((v,pi,gi))
    pairs.sort(reverse=True)
    used_p=set(); used_g=set()
    for _,pi,gi in pairs:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi); used_g.add(gi)
    return len(used_p), len(preds)-len(used_p), len(gt)-len(used_g)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='Root folder of images, e.g. /data/test/images')
    ap.add_argument('--labels', required=True, help='Root folder of matching YOLO label txt files')
    ap.add_argument('--checkpoint', required=True, help='Fine-tuned RT-DETRv2 checkpoint folder')
    ap.add_argument('--processor-id', default='PekingU/rtdetr_v2_r18vd')
    ap.add_argument('--thresholds', default='0.03,0.055,0.08')
    ap.add_argument('--nms-iou', type=float, default=0.50)
    ap.add_argument('--match-iou', type=float, default=0.50)
    ap.add_argument('--groups-csv', default=None, help='Optional CSV with image,group columns for hard-negative breakdown')
    ap.add_argument('--out', default='snakewatch_preset_evaluation')
    ap.add_argument('--max-images', type=int, default=0, help='0 evaluates all images')
    args=ap.parse_args()
    image_root=Path(args.images); label_root=Path(args.labels); ckpt=Path(args.checkpoint)
    if not image_root.exists(): raise FileNotFoundError(image_root)
    if not label_root.exists(): raise FileNotFoundError(label_root)
    if not ckpt.exists(): raise FileNotFoundError(ckpt)
    thresholds=[float(x.strip()) for x in args.thresholds.split(',') if x.strip()]
    image_paths=sorted(p for p in image_root.rglob('*') if p.suffix.lower() in IMAGE_EXTS)
    if args.max_images > 0: image_paths=image_paths[:args.max_images]
    if not image_paths: raise RuntimeError(f'No images found under {image_root}')
    group_lookup={}
    if args.groups_csv:
        gdf=pd.read_csv(args.groups_csv)
        if not {'image','group'}.issubset(gdf.columns): raise ValueError('--groups-csv needs image and group columns')
        for _,r in gdf.iterrows():
            group_lookup[str(r['image']).replace('\\','/')] = str(r['group'])
            group_lookup[Path(str(r['image'])).name] = str(r['group'])
    processor=RTDetrImageProcessor.from_pretrained(args.processor_id)
    model=RTDetrV2ForObjectDetection.from_pretrained(str(ckpt), num_labels=1, id2label={0:'snake'}, label2id={'snake':0}, ignore_mismatched_sizes=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device).eval()
    print(f'Images: {len(image_paths)} | device: {device} | thresholds: {thresholds}')
    per_image=[]; started=time.time()
    for idx,path in enumerate(image_paths,1):
        rel=path.relative_to(image_root)
        label_path=(label_root / rel).with_suffix('.txt')
        bgr=cv2.imread(str(path))
        if bgr is None: print(f'SKIP unreadable: {path}'); continue
        h,w=bgr.shape[:2]
        gt=load_yolo_labels(label_path,w,h)
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        inputs=processor(images=Image.fromarray(rgb), return_tensors='pt')
        inputs={k:v.to(device) for k,v in inputs.items()}
        with torch.inference_mode(): outputs=model(**inputs)
        # Very low post-process threshold; each candidate is thresholded again below.
        res=processor.post_process_object_detection(outputs, threshold=0.001, target_sizes=torch.tensor([[h,w]],device=device))[0]
        candidates=[]
        for score,label,box in zip(res['scores'],res['labels'],res['boxes']):
            if int(label.item()) != 0: continue
            score=float(score.item()); xyxy=[float(v) for v in box.tolist()]
            candidates.append((score,xyxy))
        group=group_lookup.get(rel.as_posix(), group_lookup.get(path.name, rel.parent.as_posix() or 'unspecified'))
        for threshold in thresholds:
            filtered=[(sc,box) for sc,box in candidates if sc >= threshold]
            if filtered:
                boxes=[b for _,b in filtered]; scores=[sc for sc,_ in filtered]
                # OpenCV NMS expects x,y,w,h
                xywh=[[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in boxes]
                keep=cv2.dnn.NMSBoxes(xywh,scores,float(threshold),float(args.nms_iou))
                pred=[boxes[int(k)] for k in np.asarray(keep).reshape(-1)] if keep is not None and len(keep) else []
            else: pred=[]
            tp,fp,fn=greedy_match(pred,gt,args.match_iou)
            per_image.append({'image':rel.as_posix(),'group':group,'threshold':threshold,'gt_count':len(gt),'pred_count':len(pred),'tp':tp,'fp':fp,'fn':fn,'image_fp':int(fp>0),'image_fn':int(fn>0)})
        if idx % 50 == 0 or idx == len(image_paths): print(f'Processed {idx}/{len(image_paths)} images')
    df=pd.DataFrame(per_image)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'per_image_results.csv',index=False)
    summary=[]
    for threshold,g in df.groupby('threshold'):
        tp=int(g.tp.sum()); fp=int(g.fp.sum()); fn=int(g.fn.sum())
        summary.append({'threshold':float(threshold),'images':int(g.image.nunique()),'GT_objects':int(g.gt_count.sum()),'predicted_objects':int(g.pred_count.sum()),'TP':tp,'FP':fp,'FN':fn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'F1':2*tp/max(2*tp+fp+fn,1),'false_positive_images':int(g.image_fp.sum()),'images_with_GT':int((g.gt_count>0).sum()),'false_positives_per_image':fp/max(g.image.nunique(),1),'matching_IoU':args.match_iou,'NMS_IoU':args.nms_iou})
    pd.DataFrame(summary).to_csv(out/'threshold_summary.csv',index=False)
    group_rows=[]
    for (threshold,group),g in df.groupby(['threshold','group']):
        group_rows.append({'threshold':float(threshold),'group':str(group),'images':int(g.image.nunique()),'GT_objects':int(g.gt_count.sum()),'TP':int(g.tp.sum()),'FP':int(g.fp.sum()),'FN':int(g.fn.sum()),'recall':int(g.tp.sum())/max(int(g.tp.sum()+g.fn.sum()),1),'false_positive_images':int(g.image_fp.sum()),'false_positives_per_image':int(g.fp.sum())/max(g.image.nunique(),1)})
    pd.DataFrame(group_rows).to_csv(out/'group_summary.csv',index=False)
    metadata={'checkpoint':str(ckpt),'processor_id':args.processor_id,'device':str(device),'thresholds':thresholds,'nms_iou':args.nms_iou,'matching_iou':args.match_iou,'images_found':len(image_paths),'images_evaluated':int(df.image.nunique()),'elapsed_seconds':round(time.time()-started,2),'notes':['Use validation data to choose a threshold; report final metrics once on a held-out test set.','Score thresholds are not calibrated probabilities.','Per-group results are only meaningful if the group labels accurately identify hard-negative categories.','Missing YOLO label files are treated as negative images; verify dataset label integrity before trusting results.']}
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('\nTHRESHOLD SUMMARY')
    print(pd.DataFrame(summary).to_string(index=False))
    print(f'\nSaved outputs to: {out.resolve()}')
if __name__=='__main__': main()
