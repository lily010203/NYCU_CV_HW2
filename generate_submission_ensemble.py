import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from Deformable_Detr import (
    DeformableDETR,
    IMAGE_SIZE_STAGE1,
    NUM_CLASSES,
    PIXEL_MEAN,
    PIXEL_STD,
    postprocess,
)
# Config
PROJECT_ROOT = Path("/home/user/NYCU-class/cv-hw2")
DATA_ROOT = PROJECT_ROOT / "nycu-hw2-data"
TEST_IMG_DIR = DATA_ROOT / "test"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_deformable_detr"

CHECKPOINT_PATHS = [
    CHECKPOINT_DIR / "epoch_4.pt",
    CHECKPOINT_DIR / "epoch_5.pt",
    CHECKPOINT_DIR / "epoch_6.pt",
]

OUTPUT_DIR = PROJECT_ROOT / "submission_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = IMAGE_SIZE_STAGE1  # (128, 320)

# TTA settings
TTA_SCALES = [1.00, 0.95, 1.05]

# WBF settings
WBF_IOU_THRESH = 0.55
WBF_SKIP_BOX_THRESH = 0.001

# NMS settings
NMS_IOU_THRESH_LIST = [0.25, 0.30, 0.35]

# submission settings
CONF_THRESHOLDS = [0.03, 0.05, 0.08]
TOP_K_LIST = [4, 5, 6]
SAVE_RAW_FUSED = False

# Utilities
def preprocess_pil(image: Image.Image) -> torch.Tensor:
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    image_tensor = (image_tensor - PIXEL_MEAN) / PIXEL_STD
    return image_tensor

def xyxy_to_xywh(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

def iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union

# TTA transforms on fixed-size input space
def apply_scale_tta(
    pil_img: Image.Image,
    scale: float,
    out_w: int,
    out_h: int,
) -> Tuple[Image.Image, Dict]:
    if abs(scale - 1.0) < 1e-6:
        return pil_img.copy(), {
            "type": "identity",
            "scale": 1.0,
            "crop_x0": 0,
            "crop_y0": 0,
            "paste_x": 0,
            "paste_y": 0,
            "new_w": out_w,
            "new_h": out_h,
        }

    new_w = max(8, int(round(out_w * scale)))
    new_h = max(8, int(round(out_h * scale)))
    scaled = pil_img.resize((new_w, new_h), Image.BILINEAR)

    if scale > 1.0:
        crop_x0 = (new_w - out_w) // 2
        crop_y0 = (new_h - out_h) // 2
        transformed = scaled.crop((crop_x0, crop_y0, crop_x0 + out_w, crop_y0 + out_h))
        meta = {
            "type": "zoom_in",
            "scale": scale,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "paste_x": 0,
            "paste_y": 0,
            "new_w": new_w,
            "new_h": new_h,
        }
    else:
        canvas = Image.new("RGB", (out_w, out_h), (114, 114, 114))
        paste_x = (out_w - new_w) // 2
        paste_y = (out_h - new_h) // 2
        canvas.paste(scaled, (paste_x, paste_y))
        transformed = canvas
        meta = {
            "type": "zoom_out",
            "scale": scale,
            "crop_x0": 0,
            "crop_y0": 0,
            "paste_x": paste_x,
            "paste_y": paste_y,
            "new_w": new_w,
            "new_h": new_h,
        }

    return transformed, meta

def invert_boxes_from_tta(
    boxes_xyxy: np.ndarray,
    meta: Dict,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return boxes_xyxy

    boxes = boxes_xyxy.copy()

    if meta["type"] == "identity":
        return boxes

    scale = meta["scale"]

    if meta["type"] == "zoom_in":
        boxes[:, [0, 2]] += meta["crop_x0"]
        boxes[:, [1, 3]] += meta["crop_y0"]
        boxes[:, [0, 2]] /= scale
        boxes[:, [1, 3]] /= scale
    elif meta["type"] == "zoom_out":
        boxes[:, [0, 2]] -= meta["paste_x"]
        boxes[:, [1, 3]] -= meta["paste_y"]
        boxes[:, [0, 2]] /= scale
        boxes[:, [1, 3]] /= scale

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, out_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, out_h - 1)
    return boxes

# WBF
def weighted_box_fusion_single_image(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_labels = sorted(set(int(l) for arr in labels_list for l in arr.tolist())) if labels_list else []
    fused_boxes = []
    fused_scores = []
    fused_labels = []

    for cls in all_labels:
        cls_boxes = []
        cls_scores = []

        for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
            if len(boxes) == 0:
                continue
            mask = labels == cls
            b = boxes[mask]
            s = scores[mask]
            for box, score in zip(b, s):
                if float(score) >= skip_box_thr:
                    cls_boxes.append(np.array(box, dtype=np.float32))
                    cls_scores.append(float(score))

        if len(cls_boxes) == 0:
            continue

        order = np.argsort(cls_scores)[::-1]
        cls_boxes = [cls_boxes[i] for i in order]
        cls_scores = [cls_scores[i] for i in order]

        clusters = []
        cluster_scores = []

        for box, score in zip(cls_boxes, cls_scores):
            matched = False
            for c_idx, cluster in enumerate(clusters):
                rep_box = np.average(cluster, axis=0, weights=cluster_scores[c_idx])
                if iou_xyxy(box, rep_box) >= iou_thr:
                    cluster.append(box)
                    cluster_scores[c_idx].append(score)
                    matched = True
                    break
            if not matched:
                clusters.append([box])
                cluster_scores.append([score])

        for cluster, scores in zip(clusters, cluster_scores):
            weights = np.array(scores, dtype=np.float32)
            boxes_np = np.stack(cluster, axis=0)
            fused_box = np.average(boxes_np, axis=0, weights=weights)
            fused_score = float(np.mean(weights))

            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
            fused_labels.append(cls)

    if len(fused_boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    fused_boxes = np.stack(fused_boxes, axis=0)
    fused_scores = np.array(fused_scores, dtype=np.float32)
    fused_labels = np.array(fused_labels, dtype=np.int64)

    order = np.argsort(fused_scores)[::-1]
    return fused_boxes[order], fused_scores[order], fused_labels[order]

# Model loading
def load_model(checkpoint_path: Path) -> torch.nn.Module:
    model = DeformableDETR(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# Single-image ensemble TTA prediction
@torch.no_grad()
def predict_single_image_ensemble_tta(
    models: List[torch.nn.Module],
    image_path: Path,
    image_id: int,
) -> Dict:
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    out_h, out_w = IMAGE_SIZE
    base_img = orig_img.resize((out_w, out_h), Image.BILINEAR)

    boxes_list = []
    scores_list = []
    labels_list = []

    for model in models:
        for scale in TTA_SCALES:
            tta_img, meta = apply_scale_tta(base_img, scale, out_w, out_h)
            image_tensor = preprocess_pil(tta_img).unsqueeze(0).to(DEVICE)

            fixed_size_tensor = torch.tensor([[out_h, out_w]], dtype=torch.long, device=DEVICE)
            outputs = model(image_tensor)
            result = postprocess(outputs, fixed_size_tensor, top_k=100)[0]

            boxes = result["boxes"].detach().cpu().numpy().astype(np.float32)
            scores = result["scores"].detach().cpu().numpy().astype(np.float32)
            labels = result["labels"].detach().cpu().numpy().astype(np.int64)

            boxes = invert_boxes_from_tta(boxes, meta, out_w, out_h)

            if len(boxes) > 0:
                boxes[:, [0, 2]] /= float(out_w)
                boxes[:, [1, 3]] /= float(out_h)
                boxes = np.clip(boxes, 0.0, 1.0)

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

    fused_boxes_norm, fused_scores, fused_labels = weighted_box_fusion_single_image(
        boxes_list=boxes_list,
        scores_list=scores_list,
        labels_list=labels_list,
        iou_thr=WBF_IOU_THRESH,
        skip_box_thr=WBF_SKIP_BOX_THRESH,
    )

    fused_boxes_fixed = fused_boxes_norm.copy()
    fused_boxes_fixed[:, [0, 2]] *= float(out_w)
    fused_boxes_fixed[:, [1, 3]] *= float(out_h)

    fused_boxes_orig = fused_boxes_fixed.copy()
    fused_boxes_orig[:, [0, 2]] *= float(orig_w) / float(out_w)
    fused_boxes_orig[:, [1, 3]] *= float(orig_h) / float(out_h)

    return {
        "image_id": image_id,
        "boxes": fused_boxes_orig.tolist(),
        "scores": fused_scores.tolist(),
        "labels": fused_labels.tolist(),
    }

# Build submission
def build_submission_from_fused(
    fused_results: List[Dict],
    conf_thresh: float,
    top_k: int,
) -> List[Dict]:
    predictions = []

    for item in fused_results:
        image_id = item["image_id"]
        boxes = np.array(item["boxes"], dtype=np.float32)
        scores = np.array(item["scores"], dtype=np.float32)
        labels = np.array(item["labels"], dtype=np.int64)

        if len(scores) == 0:
            continue

        keep = scores >= conf_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if len(scores) == 0:
            continue

        order = np.argsort(scores)[::-1]
        boxes = boxes[order][:top_k]
        scores = scores[order][:top_k]
        labels = labels[order][:top_k]

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            if x2 <= x1 or y2 <= y1:
                continue

            predictions.append(
                {
                    "image_id": int(image_id),
                    "bbox": xyxy_to_xywh([x1, y1, x2, y2]),
                    "score": float(score),
                    "category_id": int(label) + 1,
                }
            )

    return predictions

def nms_numpy_single_class(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep

def build_submission_from_fused_with_nms(
    fused_results: List[Dict],
    conf_thresh: float,
    top_k: int,
    nms_iou_thresh: float,
) -> List[Dict]:
    predictions = []

    for item in fused_results:
        image_id = item["image_id"]
        boxes = np.array(item["boxes"], dtype=np.float32)
        scores = np.array(item["scores"], dtype=np.float32)
        labels = np.array(item["labels"], dtype=np.int64)

        if len(scores) == 0:
            continue

        keep = scores >= conf_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if len(scores) == 0:
            continue

        kept_indices = []
        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_indices = np.where(cls_mask)[0]
            cls_keep_local = nms_numpy_single_class(boxes[cls_mask], scores[cls_mask], nms_iou_thresh)
            kept_indices.extend(cls_indices[cls_keep_local].tolist())

        if len(kept_indices) == 0:
            continue

        kept_indices = np.array(kept_indices, dtype=np.int64)
        boxes = boxes[kept_indices]
        scores = scores[kept_indices]
        labels = labels[kept_indices]

        order = np.argsort(scores)[::-1]
        boxes = boxes[order][:top_k]
        scores = scores[order][:top_k]
        labels = labels[order][:top_k]

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            predictions.append(
                {
                    "image_id": int(image_id),
                    "bbox": xyxy_to_xywh([x1, y1, x2, y2]),
                    "score": float(score),
                    "category_id": int(label) + 1,
                }
            )

    return predictions

# Main
def main():
    print("Device:", DEVICE)
    print("Checkpoints:")
    for ckpt in CHECKPOINT_PATHS:
        print(" -", ckpt)

    models = [load_model(path) for path in CHECKPOINT_PATHS]

    test_files = sorted(
        [p for p in TEST_IMG_DIR.iterdir() if p.suffix.lower() == ".png"],
        key=lambda p: int(p.stem),
    )
    print("Test size:", len(test_files))

    fused_results = []
    for img_path in tqdm(test_files, desc="Ensemble TTA+WBF inference"):
        image_id = int(img_path.stem)
        fused = predict_single_image_ensemble_tta(models, img_path, image_id)
        fused_results.append(fused)

    if SAVE_RAW_FUSED:
        raw_path = OUTPUT_DIR / "raw_fused_results.json"
        with open(raw_path, "w") as f:
            json.dump(fused_results, f)
        print("Saved raw fused results:", raw_path)

    for conf_thresh in CONF_THRESHOLDS:
        for top_k in TOP_K_LIST:
            preds_wbf = build_submission_from_fused(
                fused_results=fused_results,
                conf_thresh=conf_thresh,
                top_k=top_k,
            )

            out_name_wbf = f"pred_epoch456_ensemble_tta_wbf_thr{str(conf_thresh).replace('.', '')}_top{top_k}.json"
            out_path_wbf = OUTPUT_DIR / out_name_wbf
            with open(out_path_wbf, "w") as f:
                json.dump(preds_wbf, f)
            print(f"Saved WBF: {out_path_wbf} | total predictions = {len(preds_wbf)}")

            for nms_iou in NMS_IOU_THRESH_LIST:
                preds_nms = build_submission_from_fused_with_nms(
                    fused_results=fused_results,
                    conf_thresh=conf_thresh,
                    top_k=top_k,
                    nms_iou_thresh=nms_iou,
                )

                iou_tag = str(nms_iou).replace('.', '')
                out_name_nms = f"pred_epoch456_ensemble_tta_nms_iou{iou_tag}_thr{str(conf_thresh).replace('.', '')}_top{top_k}.json"
                out_path_nms = OUTPUT_DIR / out_name_nms
                with open(out_path_nms, "w") as f:
                    json.dump(preds_nms, f)
                print(f"Saved NMS: {out_path_nms} | total predictions = {len(preds_nms)}")

if __name__ == "__main__":
    main()
