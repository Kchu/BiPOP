"""Open-vocabulary detection with OWLv2 and mapping of the detections to table areas.

Reference: https://github.com/huggingface/notebooks/blob/main/examples/zeroshot_object_detection_with_owlvit.ipynb
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

MODEL_ID = "google/owlv2-base-patch16-ensemble"

# cuDNN is disabled because OWLv2 fails with cuDNN errors on some GPU/driver combinations.
torch.backends.cudnn.enabled = False


def _resize_square(image: Image.Image, size: int) -> Image.Image:
    """Bilinear resize to size x size (what transformers' former ImageFeatureExtractionMixin.resize did)."""
    return image.resize((size, size), resample=Image.BILINEAR)


def _linear_function_from_points(x1: float, y1: float, x2: float, y2: float):
    if x1 == x2:
        raise ValueError("x1 and x2 are equal, the function is not linear")
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return lambda x: slope * x + intercept


# Borders between the left/center/right table areas in normalized image
# coordinates of NICOL's left eye camera (x of the border as a function of y).
_LEFT_BORDER = _linear_function_from_points(1, 0.36, 0.5, 0.45)
_RIGHT_BORDER = _linear_function_from_points(1, 0.7, 0.5, 0.57)


def direction_recognition(text_queries: List[str], data: Dict[str, list]) -> Dict[str, str]:
    """Area ('left', 'center', 'right' or 'unknown') of the best box of every query."""
    directions = {}
    for query in text_queries:
        _, bbox = data[query]
        if bbox is None:  # not detected above the score threshold
            directions[query] = "unknown"
            continue
        cx, cy, _, _ = bbox
        if cy < 0.4:
            if cx < 0.466:
                directions[query] = "left"
            elif cx > 0.557:
                directions[query] = "right"
            else:
                directions[query] = "center"
        elif cx < _LEFT_BORDER(cy):
            directions[query] = "left"
        elif cx > _RIGHT_BORDER(cy):
            directions[query] = "right"
        else:
            directions[query] = "center"
    return directions


def owlv2_solve(
    image,
    text_queries: List[str],
    threshold: float,
    save_path: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, list]]:
    """Detect ``text_queries`` in a PIL image.

    Returns the table area of every query and the best box ``[cx, cy, w, h]``
    (normalized) of every detected query. With ``save_path`` the detections
    are drawn and saved as an image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Owlv2Processor.from_pretrained(MODEL_ID, use_fast=True)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(device)
    model.eval()

    if image.height != image.width:
        image = _resize_square(image, min(image.height, image.width))

    inputs = processor(text=text_queries, images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    logits = torch.max(outputs["logits"][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()
    labels = logits.indices.cpu().detach().numpy()
    boxes = outputs["pred_boxes"][0].cpu().detach().numpy()

    # Keep the highest-scoring box per query.
    best: Dict[str, list] = {query: [0, None] for query in text_queries}
    detections: Dict[str, list] = {}
    for score, box, label in zip(scores, boxes, labels):
        if score < threshold:
            continue
        query = text_queries[label]
        if score > best[query][0]:
            best[query] = [score, list(box)]
            detections[query] = list(box)

    directions = direction_recognition(text_queries, best)
    print("OWLv2 directions:", directions)

    if save_path:
        _save_visualization(_resize_square(image, model.config.vision_config.image_size), best, save_path)
    return directions, detections


def _save_visualization(image, best: Dict[str, list], save_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(np.asarray(image).astype(np.float32) / 255.0, extent=(0, 1, 1, 0))
    ax.set_axis_off()
    for label, (score, bbox) in best.items():
        if bbox is None:
            continue
        cx, cy, w, h = bbox
        ax.plot([cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2], "r")
        ax.text(cx - w / 2, cy + h / 2 + 0.015, f"{label}: {score:1.2f}", ha="left", va="top", color="red",
                fontsize=14, bbox={"facecolor": "white", "edgecolor": "red", "boxstyle": "square,pad=.3"})
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
