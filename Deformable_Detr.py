import os
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import nms
from tqdm.auto import tqdm

# Config: v6 single-stage = 4-level feature + DAB-style 4D reference
PROJECT_ROOT = Path("/home/user/NYCU-class/cv-hw2")
DATA_ROOT = PROJECT_ROOT / "nycu-hw2-data"

TRAIN_IMG_DIR = DATA_ROOT / "train"
VALID_IMG_DIR = DATA_ROOT / "valid"
TEST_IMG_DIR = DATA_ROOT / "test"
TRAIN_JSON = DATA_ROOT / "train.json"
VALID_JSON = DATA_ROOT / "valid.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs_deformable_detr"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_deformable_detr"
VALID_PRED_DIR = OUTPUT_DIR / "valid_preds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
VALID_PRED_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
LOG_TXT_PATH = OUTPUT_DIR / "training_log.txt"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 10
NO_OBJECT_CLASS = NUM_CLASSES

NUM_EPOCHS = 8
START_EPOCH = 1
IMAGE_SIZE = (128, 320)
BATCH_SIZE = 8
NUM_WORKERS = 4

BACKBONE_LR = 1e-5
OTHER_LR = 1e-4
WEIGHT_DECAY = 1e-4
MIN_LR_RATIO = 0.10
EOS_COEF = 0.1

MAX_DETECTIONS_PER_IMAGE = 100
CONF_THRESHOLDS = [0.03, 0.05, 0.08, 0.10]
TOP_K_PER_IMAGE = 10

VAL_CONF_THRESH = 0.05
VAL_TOP_K = 5
VAL_USE_NMS = True
VAL_NMS_IOU = 0.3

NUM_QUERIES = 100
HIDDEN_DIM = 256
NHEAD = 8
ENC_LAYERS = 4
DEC_LAYERS = 4
FFN_DIM = 1024
DROPOUT = 0.1
NUM_FEATURE_LEVELS = 4  # C2 / C3 / C4 / C5
NUM_POINTS = 4
AUX_LOSS = True

COST_CLASS = 2.0
COST_BBOX = 5.0
COST_GIOU = 2.0
LOSS_CLASS = 2.0
LOSS_BBOX = 5.0
LOSS_GIOU = 2.0

GRAD_CLIP = 0.1
SAVE_DEBUG_IDS = [2620, 457, 103]

PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
PIXEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Utilities
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def append_log_line(text: str) -> None:
    with open(LOG_TXT_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def init_log_file() -> None:
    with open(LOG_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("Deformable DETR v6 single-stage training log\n")
        f.write("4-level feature + DAB-style 4D anchors\n")
        f.write(f"PROJECT_ROOT={PROJECT_ROOT}\n")
        f.write(f"DATA_ROOT={DATA_ROOT}\n")
        f.write(f"DEVICE={DEVICE}\n")
        f.write(f"NUM_EPOCHS={NUM_EPOCHS}\n")
        f.write(f"IMAGE_SIZE={IMAGE_SIZE}\n")
        f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
        f.write(f"BACKBONE_LR={BACKBONE_LR}\n")
        f.write(f"OTHER_LR={OTHER_LR}\n")
        f.write("=" * 80 + "\n")

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)

def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    lt_c = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-6)

def bias_init_with_prob(prior_prob: float) -> float:
    return float(-math.log((1 - prior_prob) / prior_prob))

def denormalize_image(x: torch.Tensor) -> torch.Tensor:
    return (x * PIXEL_STD.to(x.device) + PIXEL_MEAN.to(x.device)).clamp(0, 1)

def get_scale_xyxy_from_target(target: Dict, device: torch.device) -> torch.Tensor:
    h, w = target["size"].tolist()
    return torch.tensor([w, h, w, h], device=device, dtype=torch.float32)

# Dataset
class DigitCocoDataset(Dataset):
    def __init__(self, image_dir: Path, annotation_json: Path, fixed_size: Tuple[int, int]):
        self.image_dir = Path(image_dir)
        self.coco = load_json(annotation_json)
        self.images = self.coco["images"]
        self.fixed_h, self.fixed_w = fixed_size
        self.ann_by_img = {}
        for ann in self.coco["annotations"]:
            self.ann_by_img.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def _resize_with_boxes(self, image: Image.Image, anns: List[Dict]) -> Tuple[Image.Image, Dict]:
        orig_w, orig_h = image.size
        new_h, new_w = self.fixed_h, self.fixed_w
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        image = image.resize((new_w, new_h), Image.BILINEAR)

        boxes = []
        labels = []
        areas = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"] - 1)
            areas.append((x2 - x1) * (y2 - y1))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "size": torch.tensor([new_h, new_w], dtype=torch.long),
        }
        return image, target

    def __getitem__(self, idx: int) -> Dict:
        img_info = self.images[idx]
        image_id = img_info["id"]
        file_name = img_info["file_name"]
        image = Image.open(self.image_dir / file_name).convert("RGB")
        anns = self.ann_by_img.get(image_id, [])
        image, target = self._resize_with_boxes(image, anns)
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - PIXEL_MEAN) / PIXEL_STD
        return {"image": image_tensor, "target": target, "image_id": image_id, "file_name": file_name}

class DigitTestDataset(Dataset):
    def __init__(self, image_dir: Path, fixed_size: Tuple[int, int]):
        self.image_dir = Path(image_dir)
        self.fixed_h, self.fixed_w = fixed_size
        self.files = sorted([p.name for p in self.image_dir.iterdir() if p.suffix.lower() == ".png"], key=lambda x: int(Path(x).stem))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        file_name = self.files[idx]
        image = Image.open(self.image_dir / file_name).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.fixed_w, self.fixed_h), Image.BILINEAR)
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - PIXEL_MEAN) / PIXEL_STD
        return {
            "image": image_tensor,
            "image_id": int(Path(file_name).stem),
            "file_name": file_name,
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "size": torch.tensor([self.fixed_h, self.fixed_w], dtype=torch.long),
        }

def collate_fn(batch: List[Dict]) -> Dict:
    result = {
        "images": torch.stack([x["image"] for x in batch], dim=0),
        "image_ids": [x["image_id"] for x in batch],
        "file_names": [x["file_name"] for x in batch],
    }
    if "target" in batch[0]:
        result["targets"] = [x["target"] for x in batch]
        result["orig_sizes"] = torch.stack([x["target"]["orig_size"] for x in batch], dim=0)
        result["sizes"] = torch.stack([x["target"]["size"] for x in batch], dim=0)
    else:
        result["orig_sizes"] = torch.stack([x["orig_size"] for x in batch], dim=0)
        result["sizes"] = torch.stack([x["size"] for x in batch], dim=0)
    return result

def build_dataloaders():
    train_dataset = DigitCocoDataset(TRAIN_IMG_DIR, TRAIN_JSON, IMAGE_SIZE)
    valid_dataset = DigitCocoDataset(VALID_IMG_DIR, VALID_JSON, IMAGE_SIZE)
    test_dataset = DigitTestDataset(TEST_IMG_DIR, IMAGE_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader

# Hungarian Matcher
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 1.0, cost_giou: float = 1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]):
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]
        indices = []
        for b in range(outputs["pred_logits"].shape[0]):
            tgt_ids = targets[b]["labels"]
            if tgt_ids.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            scale_xyxy = get_scale_xyxy_from_target(targets[b], out_bbox.device)
            tgt_bbox = box_xyxy_to_cxcywh(targets[b]["boxes"] / scale_xyxy)
            cost_class = -out_prob[b][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[b]), box_cxcywh_to_xyxy(tgt_bbox))
            cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append((torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)))
        return indices

# Backbone
class BackboneResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]

# Position Encoding
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True, scale: float = 2 * math.pi):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

# Simplified MS Deformable Attention
class MSDeformAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_levels: int, n_points: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.d_per_head = d_model // n_heads
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query: torch.Tensor, reference_points: torch.Tensor, srcs: List[torch.Tensor]) -> torch.Tensor:
        b, num_query, _ = query.shape
        value = [src.flatten(2).transpose(1, 2) for src in srcs]
        value = torch.cat(value, dim=1)
        value = self.value_proj(value)
        value = value.view(b, value.shape[1], self.n_heads, self.d_per_head)

        sampling_offsets = self.sampling_offsets(query).view(b, num_query, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(b, num_query, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(b, num_query, self.n_heads, self.n_levels, self.n_points)

        split_sizes = [src.shape[2] * src.shape[3] for src in srcs]
        value_list = value.split(split_sizes, dim=1)
        output = torch.zeros((b, num_query, self.n_heads, self.d_per_head), device=query.device)

        for lvl, src in enumerate(srcs):
            _, _, h, w = src.shape
            value_l = value_list[lvl].view(b, h * w, self.n_heads, self.d_per_head)
            value_l = value_l.permute(0, 2, 3, 1).contiguous().view(b * self.n_heads, self.d_per_head, h, w)
            ref = reference_points[:, :, None, None, lvl:lvl + 1, :].repeat(1, 1, self.n_heads, self.n_points, 1, 1)
            ref = ref.squeeze(4)
            offsets = sampling_offsets[:, :, :, lvl]
            normalizer = torch.tensor([w, h], dtype=torch.float32, device=query.device)
            sampling_locations = ref + offsets / normalizer
            grid = sampling_locations * 2.0 - 1.0
            grid = grid.permute(0, 2, 1, 3, 4).contiguous().view(b * self.n_heads, num_query * self.n_points, 1, 2)
            sampled = F.grid_sample(value_l, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            sampled = sampled.view(b, self.n_heads, self.d_per_head, num_query, self.n_points)
            sampled = sampled.permute(0, 3, 1, 4, 2)
            attn = attention_weights[:, :, :, lvl].unsqueeze(-1)
            output = output + (sampled * attn).sum(dim=3)

        output = output.view(b, num_query, self.d_model)
        return self.output_proj(output)

# Transformer Layers
class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float, n_heads: int, n_levels: int, n_points: int):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_heads, n_levels, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, reference_points: torch.Tensor, srcs_2d: List[torch.Tensor]) -> torch.Tensor:
        src2 = self.self_attn(src, reference_points, srcs_2d)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout3(src2))
        return src

class DeformableDecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float, n_heads: int, n_levels: int, n_points: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn(d_model, n_heads, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, query_pos: torch.Tensor, reference_points_xy: torch.Tensor, srcs_2d: List[torch.Tensor]) -> torch.Tensor:
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.cross_attn(tgt + query_pos, reference_points_xy, srcs_2d)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout3(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        return tgt

# Model: DAB-style 4D anchor references
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

class DeformableDETR(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = BackboneResNet50()
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, HIDDEN_DIM, kernel_size=1), nn.GroupNorm(32, HIDDEN_DIM)),
            nn.Sequential(nn.Conv2d(512, HIDDEN_DIM, kernel_size=1), nn.GroupNorm(32, HIDDEN_DIM)),
            nn.Sequential(nn.Conv2d(1024, HIDDEN_DIM, kernel_size=1), nn.GroupNorm(32, HIDDEN_DIM)),
            nn.Sequential(nn.Conv2d(2048, HIDDEN_DIM, kernel_size=1), nn.GroupNorm(32, HIDDEN_DIM)),
        ])
        self.position_embedding = PositionEmbeddingSine(HIDDEN_DIM // 2)
        self.encoder_layers = nn.ModuleList([DeformableEncoderLayer(HIDDEN_DIM, FFN_DIM, DROPOUT, NHEAD, NUM_FEATURE_LEVELS, NUM_POINTS) for _ in range(ENC_LAYERS)])
        self.decoder_layers = nn.ModuleList([DeformableDecoderLayer(HIDDEN_DIM, FFN_DIM, DROPOUT, NHEAD, NUM_FEATURE_LEVELS, NUM_POINTS) for _ in range(DEC_LAYERS)])

        self.level_embed = nn.Parameter(torch.Tensor(NUM_FEATURE_LEVELS, HIDDEN_DIM))
        self.query_embed = nn.Embedding(NUM_QUERIES, HIDDEN_DIM)
        self.query_ref = nn.Embedding(NUM_QUERIES, 4)

        self.class_embed = nn.ModuleList([nn.Linear(HIDDEN_DIM, num_classes + 1) for _ in range(DEC_LAYERS)])
        self.bbox_embed = nn.ModuleList([MLP(HIDDEN_DIM, HIDDEN_DIM, 4, 3) for _ in range(DEC_LAYERS)])
        self.enc_output = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.enc_output_norm = nn.LayerNorm(HIDDEN_DIM)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.level_embed)
        nn.init.xavier_uniform_(self.query_embed.weight)
        nn.init.uniform_(self.query_ref.weight[:, :2], 0, 1)
        nn.init.constant_(self.query_ref.weight[:, 2:], -2.0)
        for cls_head in self.class_embed:
            nn.init.constant_(cls_head.bias, bias_init_with_prob(0.01))
            cls_head.bias.data[NO_OBJECT_CLASS] = 0.0
        for bbox_head in self.bbox_embed:
            nn.init.constant_(bbox_head.layers[-1].weight.data, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias.data, 0.0)

    def _get_reference_points_for_encoder(self, srcs: List[torch.Tensor]) -> torch.Tensor:
        all_refs = []
        for src in srcs:
            b, _, h, w = src.shape
            y, x = torch.meshgrid(
                torch.linspace(0.5 / h, 1 - 0.5 / h, h, device=src.device),
                torch.linspace(0.5 / w, 1 - 0.5 / w, w, device=src.device),
                indexing="ij",
            )
            ref = torch.stack((x, y), dim=-1).view(1, h * w, 1, 2).repeat(b, 1, NUM_FEATURE_LEVELS, 1)
            all_refs.append(ref)
        return torch.cat(all_refs, dim=1)

    @staticmethod
    def _split_memory_by_levels(memory: torch.Tensor, srcs: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        start = 0
        for src in srcs:
            b, c, h, w = src.shape
            length = h * w
            mem_l = memory[:, start:start + length, :]
            mem_l = mem_l.transpose(1, 2).contiguous().view(b, c, h, w)
            outputs.append(mem_l)
            start += length
        return outputs

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(images)
        srcs = [proj(feat) for proj, feat in zip(self.input_proj, feats)]
        pos_embeds = [self.position_embedding(src) + self.level_embed[i].view(1, -1, 1, 1) for i, src in enumerate(srcs)]

        src_flatten = []
        srcs_2d = []
        for src, pos in zip(srcs, pos_embeds):
            src = src + pos
            srcs_2d.append(src)
            src_flatten.append(src.flatten(2).transpose(1, 2))
        memory = torch.cat(src_flatten, dim=1)

        enc_ref = self._get_reference_points_for_encoder(srcs)
        for layer in self.encoder_layers:
            memory = layer(memory, enc_ref, srcs_2d)
        memory = self.enc_output_norm(self.enc_output(memory))
        memory_2d = self._split_memory_by_levels(memory, srcs)

        bs = images.shape[0]
        tgt = torch.zeros((bs, NUM_QUERIES, HIDDEN_DIM), device=images.device)
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        reference_boxes = self.query_ref.weight.sigmoid().unsqueeze(0).repeat(bs, 1, 1)

        outputs_classes = []
        outputs_coords = []
        for lid, layer in enumerate(self.decoder_layers):
            reference_points_xy = reference_boxes[..., :2].unsqueeze(2).repeat(1, 1, NUM_FEATURE_LEVELS, 1)
            tgt = layer(tgt, query_pos, reference_points_xy, memory_2d)
            outputs_class = self.class_embed[lid](tgt)
            delta_box = self.bbox_embed[lid](tgt)
            outputs_coord = (delta_box + inverse_sigmoid(reference_boxes)).sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            reference_boxes = outputs_coord.detach()

        out = {"pred_logits": outputs_classes[-1], "pred_boxes": outputs_coords[-1]}
        if AUX_LOSS:
            out["aux_outputs"] = [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_classes[:-1], outputs_coords[:-1])]
        return out

# Criterion
class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: HungarianMatcher, eos_coef: float = EOS_COEF):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"]
        bs, nq, _ = src_logits.shape
        target_classes = torch.full((bs, nq), NO_OBJECT_CLASS, dtype=torch.long, device=src_logits.device)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)
        return {"loss_cls": loss_ce}

    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        if len(idx[0]) == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = []
        for t, (_, i) in zip(targets, indices):
            scale_xyxy = get_scale_xyxy_from_target(t, src_boxes.device)
            target_boxes.append(box_xyxy_to_cxcywh(t["boxes"][i] / scale_xyxy))
        target_boxes = torch.cat(target_boxes, dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / max(target_boxes.shape[0], 1)
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou.sum() / max(target_boxes.shape[0], 1)
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs, targets):
        outputs_wo_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_wo_aux, targets)
        losses = {}
        losses.update(self.loss_labels(outputs_wo_aux, targets, indices))
        losses.update(self.loss_boxes(outputs_wo_aux, targets, indices))
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                aux_losses = {}
                aux_losses.update(self.loss_labels(aux_outputs, targets, aux_indices))
                aux_losses.update(self.loss_boxes(aux_outputs, targets, aux_indices))
                for k, v in aux_losses.items():
                    losses[f"{k}_{i}"] = v
        total = 0.0
        for k, v in losses.items():
            if "loss_cls" in k:
                total += LOSS_CLASS * v
            elif "loss_bbox" in k:
                total += LOSS_BBOX * v
            elif "loss_giou" in k:
                total += LOSS_GIOU * v
        losses["loss_total"] = total
        return losses

# Postprocess / Eval
@torch.no_grad()
def postprocess(outputs: Dict[str, torch.Tensor], orig_sizes: torch.Tensor, top_k: int = 100):
    prob = outputs["pred_logits"].softmax(-1)
    boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"])
    results = []
    for i in range(prob.shape[0]):
        scores_all = prob[i][:, :NUM_CLASSES]
        scores, labels = scores_all.max(dim=-1)
        no_obj_scores = prob[i][:, NO_OBJECT_CLASS]
        keep = (labels >= 0) & (scores > no_obj_scores)
        scores = scores[keep]
        labels = labels[keep]
        boxes_i = boxes[i][keep]
        if scores.numel() > 0:
            topk = min(top_k, scores.shape[0])
            topk_scores, topk_idx = torch.topk(scores, k=topk)
            labels = labels[topk_idx]
            boxes_i = boxes_i[topk_idx]
            orig_h, orig_w = orig_sizes[i]
            scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=torch.float32, device=boxes_i.device)
            boxes_i = boxes_i * scale
        else:
            topk_scores = torch.zeros((0,), device=prob.device)
            labels = torch.zeros((0,), dtype=torch.long, device=prob.device)
            boxes_i = torch.zeros((0, 4), device=prob.device)
        results.append({"scores": topk_scores, "labels": labels, "boxes": boxes_i})
    return results

def apply_nms_per_image(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_threshold: float = 0.3):
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep_all = []
    for cls in labels.unique():
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = torch.where(cls_mask)[0]
        keep = nms(cls_boxes, cls_scores, iou_threshold)
        keep_all.append(cls_indices[keep])
    if len(keep_all) == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
    keep_all = torch.cat(keep_all)
    keep_all = keep_all[scores[keep_all].argsort(descending=True)]
    return boxes[keep_all], scores[keep_all], labels[keep_all]

@torch.no_grad()
def collect_valid_predictions(model, loader):
    model.eval()
    raw_results = []
    for batch in tqdm(loader, desc="Collect valid preds", leave=False):
        images = batch["images"].to(DEVICE, non_blocking=True)
        orig_sizes = batch["orig_sizes"].to(DEVICE)
        outputs = model(images)
        results = postprocess(outputs, orig_sizes, top_k=100)
        for image_id, result in zip(batch["image_ids"], results):
            raw_results.append({
                "image_id": int(image_id),
                "boxes": result["boxes"].detach().cpu(),
                "scores": result["scores"].detach().cpu(),
                "labels": result["labels"].detach().cpu(),
            })
    return raw_results

def build_valid_predictions_for_eval(raw_results, conf_thresh=0.05, top_k=5, use_nms=True, nms_iou=0.3):
    predictions = []
    for item in raw_results:
        image_id = item["image_id"]
        boxes = item["boxes"]
        scores = item["scores"]
        labels = item["labels"]
        keep = scores >= conf_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if use_nms:
            boxes, scores, labels = apply_nms_per_image(boxes, scores, labels, nms_iou)
        if scores.numel() > 0:
            order = scores.argsort(descending=True)
            boxes = boxes[order][:top_k]
            scores = scores[order][:top_k]
            labels = labels[order][:top_k]
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            predictions.append({
                "image_id": int(image_id),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score),
                "category_id": int(label) + 1,
            })
    return predictions

def evaluate_valid_map_from_json(gt_json_path: Path, pred_json_path: Path):
    coco_gt = COCO(str(gt_json_path))
    coco_dt = coco_gt.loadRes(str(pred_json_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    return {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
    }

@torch.no_grad()
def evaluate_valid_map(model, loader, epoch: int):
    raw_results = collect_valid_predictions(model, loader)
    preds = build_valid_predictions_for_eval(raw_results, VAL_CONF_THRESH, VAL_TOP_K, VAL_USE_NMS, VAL_NMS_IOU)
    pred_path = VALID_PRED_DIR / f"valid_epoch_{epoch}.json"
    with open(pred_path, "w") as f:
        json.dump(preds, f)
    metrics = evaluate_valid_map_from_json(VALID_JSON, pred_path)
    metrics["num_predictions"] = len(preds)
    metrics["pred_json_path"] = str(pred_path)
    return metrics

# Train helpers
def build_optimizer_and_scheduler(model: nn.Module, train_loader: DataLoader):
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
        {"params": other_params, "lr": OTHER_LR, "weight_decay": WEIGHT_DECAY},
    ])

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = min(len(train_loader), 1000)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return max(1e-3, float(current_step) / float(max(1, warmup_steps)))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(MIN_LR_RATIO, cosine)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler

def save_checkpoint(path: Path, epoch: int, model: nn.Module, optimizer, scheduler, best_valid_loss: float, best_valid_map_50_95: float, train_loss: float, valid_loss: float, valid_metrics: Dict):
    torch.save({
        "epoch": epoch,
        "image_size": IMAGE_SIZE,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_valid_loss": best_valid_loss,
        "best_valid_map_50_95": best_valid_map_50_95,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "valid_metrics": valid_metrics,
    }, path)

@torch.no_grad()
def save_debug_predictions(model: nn.Module, dataset: DigitCocoDataset, epoch: int):
    debug_dir = OUTPUT_DIR / "valid_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    id_to_index = {img["id"]: i for i, img in enumerate(dataset.images)}
    for image_id in SAVE_DEBUG_IDS:
        if image_id not in id_to_index:
            continue
        sample = dataset[id_to_index[image_id]]
        image = sample["image"].unsqueeze(0).to(DEVICE)
        outputs = model(image)
        h, w = sample["target"]["size"].tolist()
        debug_size = torch.tensor([[h, w]], device=DEVICE)
        result = postprocess(outputs, debug_size, top_k=20)[0]
        image_vis = denormalize_image(sample["image"]).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        pil_img = Image.fromarray(image_vis)
        draw = ImageDraw.Draw(pil_img)
        for box, label in zip(sample["target"]["boxes"], sample["target"]["labels"]):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, max(0, y1 - 12)), f"GT:{int(label) + 1}", fill="green")
        kept = 0
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            if score.item() < 0.05:
                continue
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, max(0, y1 - 12)), f"P:{int(label) + 1}:{score.item():.2f}", fill="red")
            kept += 1
        out_path = debug_dir / f"epoch_{epoch}_img_{image_id}.png"
        pil_img.save(out_path)
        msg = f"[Epoch {epoch}] saved debug -> {out_path} | kept={kept} | gt={len(sample['target']['labels'])}"
        print(msg)
        append_log_line(msg)

# Train / Valid / Infer
def train_one_epoch(model, criterion, loader, optimizer, scheduler, epoch):
    model.train()
    criterion.train()
    running = 0.0
    valid_steps = 0
    pbar = tqdm(loader, desc=f"Train {epoch}")
    for batch in pbar:
        images = batch["images"].to(DEVICE, non_blocking=True)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["targets"]]
        optimizer.zero_grad()
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["loss_total"]
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        running += loss.item()
        valid_steps += 1
        pbar.set_postfix({
            "loss": f"{running / max(valid_steps, 1):.4f}",
            "bb_lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "oth_lr": f"{optimizer.param_groups[1]['lr']:.2e}",
        })
    return running / max(valid_steps, 1)

@torch.no_grad()
def validate_one_epoch(model, criterion, loader, epoch):
    model.eval()
    criterion.eval()
    running = 0.0
    valid_steps = 0
    pbar = tqdm(loader, desc=f"Valid {epoch}")
    for batch in pbar:
        images = batch["images"].to(DEVICE, non_blocking=True)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["targets"]]
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["loss_total"]
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        running += loss.item()
        valid_steps += 1
        pbar.set_postfix({"loss": f"{running / max(valid_steps, 1):.4f}"})
    return running / max(valid_steps, 1)

@torch.no_grad()
def generate_submissions(model, loader):
    model.eval()
    raw_results = []
    for batch in tqdm(loader, desc="Inference"):
        images = batch["images"].to(DEVICE, non_blocking=True)
        orig_sizes = batch["orig_sizes"].to(DEVICE)
        outputs = model(images)
        results = postprocess(outputs, orig_sizes, top_k=MAX_DETECTIONS_PER_IMAGE)
        for image_id, result in zip(batch["image_ids"], results):
            raw_results.append({
                "image_id": int(image_id),
                "boxes": result["boxes"].cpu().tolist(),
                "scores": result["scores"].cpu().tolist(),
                "labels": result["labels"].cpu().tolist(),
            })
    for thresh in CONF_THRESHOLDS:
        preds = []
        for item in raw_results:
            filtered = []
            for box, score, label in zip(item["boxes"], item["scores"], item["labels"]):
                if score < thresh:
                    continue
                filtered.append((box, score, label))
            filtered = filtered[:TOP_K_PER_IMAGE]
            for box, score, label in filtered:
                x1, y1, x2, y2 = box
                preds.append({
                    "image_id": item["image_id"],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                    "category_id": int(label) + 1,
                })
        out_path = OUTPUT_DIR / f"pred_thresh{str(thresh).replace('.', '')}_top{TOP_K_PER_IMAGE}.json"
        with open(out_path, "w") as f:
            json.dump(preds, f)
        msg = f"Saved {out_path} | total predictions = {len(preds)}"
        print(msg)
        append_log_line(msg)

# Main
def main():
    set_seed(SEED)
    init_log_file()
    print("Device:", DEVICE)
    append_log_line("Start training")
    append_log_line(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print("GPU:", gpu_name)
        append_log_line(f"GPU: {gpu_name}")

    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = build_dataloaders()
    append_log_line(f"Train size: {len(train_dataset)} | Valid size: {len(valid_dataset)} | Test size: {len(test_dataset)}")

    model = DeformableDETR(num_classes=NUM_CLASSES).to(DEVICE)
    matcher = HungarianMatcher(COST_CLASS, COST_BBOX, COST_GIOU)
    criterion = SetCriterion(NUM_CLASSES, matcher, eos_coef=EOS_COEF).to(DEVICE)
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader)

    best_valid_loss = float("inf")
    best_valid_map_50_95 = float("-inf")

    for epoch in range(START_EPOCH, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, epoch)
        valid_loss = validate_one_epoch(model, criterion, valid_loader, epoch)
        valid_metrics = evaluate_valid_map(model, valid_loader, epoch)
        current_map_50_95 = valid_metrics["mAP_50_95"]

        msg = (
            f"[Epoch {epoch}] train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
            f"valid_mAP@0.5:0.95={current_map_50_95:.6f} valid_mAP@0.5={valid_metrics['mAP_50']:.6f}"
        )
        print(msg)
        append_log_line(
            f"[Epoch {epoch}] train_loss={train_loss:.6f} | valid_loss={valid_loss:.6f} | "
            f"valid_mAP@0.5:0.95={valid_metrics['mAP_50_95']:.6f} | valid_mAP@0.5={valid_metrics['mAP_50']:.6f} | "
            f"valid_mAP@0.75={valid_metrics['mAP_75']:.6f} | mAP_small={valid_metrics['mAP_small']:.6f} | "
            f"AR@1={valid_metrics['AR_1']:.6f} | AR@10={valid_metrics['AR_10']:.6f} | AR@100={valid_metrics['AR_100']:.6f} | num_preds={valid_metrics['num_predictions']}"
        )

        save_checkpoint(
            CHECKPOINT_DIR / f"epoch_{epoch}.pt",
            epoch,
            model,
            optimizer,
            scheduler,
            best_valid_loss,
            best_valid_map_50_95,
            train_loss,
            valid_loss,
            valid_metrics,
        )
        save_debug_predictions(model, valid_dataset, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

        if current_map_50_95 > best_valid_map_50_95:
            best_valid_map_50_95 = current_map_50_95
            save_checkpoint(
                BEST_MODEL_PATH,
                epoch,
                model,
                optimizer,
                scheduler,
                best_valid_loss,
                best_valid_map_50_95,
                train_loss,
                valid_loss,
                valid_metrics,
            )
            best_msg = (
                f"Saved best checkpoint by mAP -> {BEST_MODEL_PATH} | epoch={epoch} | "
                f"image_size={IMAGE_SIZE} | best_mAP@0.5:0.95={best_valid_map_50_95:.6f}"
            )
            print(best_msg)
            append_log_line(best_msg)

        append_log_line("-" * 80)

    best_ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    print("Loaded best checkpoint for inference.")
    append_log_line(
        f"Loaded best checkpoint for inference | epoch={best_ckpt.get('epoch', 'unknown')} | "
        f"image_size={best_ckpt.get('image_size', 'unknown')} | "
        f"best_valid_map_50_95={best_ckpt.get('best_valid_map_50_95', 'unknown')}"
    )
    generate_submissions(model, test_loader)
    append_log_line("Finished inference and submission generation")

if __name__ == "__main__":
    main()
