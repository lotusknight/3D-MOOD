"""Demo 3D-MOOD with custom images."""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vis4d.data.transforms.base import compose
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.resize import ResizeImages, ResizeIntrinsics
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.common.ckpt import load_model_checkpoint
from vis4d.op.fpp.fpn import FPN
from vis4d.vis.image.functional import imshow_bboxes3d

from opendet3d.data.transforms.pad import CenterPadImages, CenterPadIntrinsics
from opendet3d.data.transforms.resize import GenResizeParameters
from opendet3d.model.detect3d.grounding_dino_3d import GroundingDINO3D
from opendet3d.op.base.swin import SwinTransformer
from opendet3d.op.detect3d.grounding_dino_3d import (
    GroundingDINO3DCoder,
    GroundingDINO3DHead,
    RoI2Det3D,
    UniDepthHead,
)
from opendet3d.op.fpp.channel_mapper import ChannelMapper


def get_3d_mood_swin_base(
    max_per_image: int = 100,
    score_thres: float = 0.1,
    nms: bool = True,
    class_agnostic_nms: bool = True,
    iou_thres: float = 0.5,
) -> GroundingDINO3D:
    """Get the config of Swin-Base."""
    basemodel = SwinTransformer(
        convert_weights=True,
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.3,
        out_indices=(0, 1, 2, 3),
    )

    neck = ChannelMapper(
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm="GroupNorm",
        num_groups=32,
        activation=None,
        bias=True,
    )

    depth_fpn = FPN(
        in_channels_list=[128, 256, 512, 1024],
        out_channels=256,
        extra_blocks=None,
        start_index=0,
    )

    depth_head = UniDepthHead(input_dims=[256, 256, 256, 256])

    box_coder = GroundingDINO3DCoder()

    bbox3d_head = GroundingDINO3DHead(box_coder=box_coder)

    roi2det3d = RoI2Det3D(
        nms=nms,
        class_agnostic_nms=class_agnostic_nms,
        max_per_img=max_per_image,
        score_threshold=score_thres,
        iou_threshold=iou_thres,
    )

    return GroundingDINO3D(
        basemodel=basemodel,
        neck=neck,
        bbox3d_head=bbox3d_head,
        roi2det3d=roi2det3d,
        fpn=depth_fpn,
        depth_head=depth_head,
    )


def _save_input_output_preview(
    input_path: Path, output_path: Path, save_path: Path
) -> None:
    """Save a side-by-side RGB preview (input | output), height-aligned."""
    left = Image.open(input_path).convert("RGB")
    right = Image.open(output_path).convert("RGB")
    if right.height != left.height:
        new_w = max(1, int(round(right.width * left.height / right.height)))
        right = right.resize((new_w, left.height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (left.width + right.width, left.height), (40, 40, 40))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(save_path)


if __name__ == "__main__":
    """Demo."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo_dir = Path(__file__).resolve().parent.parent / "assets" / "demo"
    input_image_path = demo_dir / "rgb.png"
    output_image_path = demo_dir / "output.png"
    compare_image_path = demo_dir / "input_output_compare.png"

    text_prompts = "chair.table"

    # Data
    images = np.array(Image.open(input_image_path)).astype(np.float32)[
        None, ...
    ]
    intrinsics = np.load("./assets/demo/intrinsics.npy")

    input_texts = text_prompts.split(".")
    class_id_mapping = {i: text for i, text in enumerate(input_texts)}

    data_dict = {
        "images": images,
        "original_images": images,
        "input_hw": (images.shape[1], images.shape[2]),
        "original_hw": (images.shape[1], images.shape[2]),
        "intrinsics": intrinsics,
        "original_intrinsics": intrinsics,
    }

    # Transform
    preprocess_transforms = compose(
        transforms=[
            GenResizeParameters(shape=(800, 1333)),
            ResizeImages(),
            ResizeIntrinsics(),
            NormalizeImages(),
            CenterPadImages(stride=1, shape=(800, 1333), update_input_hw=True),
            CenterPadIntrinsics(),
        ]
    )

    data = preprocess_transforms([data_dict])[0]

    # Convert to Tensor
    to_tensor = ToTensor()
    data = to_tensor([data])[0]

    # Model build (structure + move to device)
    t0 = time.perf_counter()
    model = get_3d_mood_swin_base().to(device)
    t1 = time.perf_counter()
    model_build_s = t1 - t0

    # Checkpoint load
    t0 = time.perf_counter()
    load_model_checkpoint(
        model,
        weights="https://huggingface.co/RoyYang0714/3D-MOOD/resolve/main/gdino3d_swin-b_120e_omni3d_834c97.pt",
        rev_keys=[(r"^model\.", ""), (r"^module\.", "")],
    )
    t1 = time.perf_counter()
    checkpoint_load_s = t1 - t0

    model.eval()

    # Inference (forward only)
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        boxes, boxes3d, scores, class_ids, depth_maps, categories = model(
            images=data["images"].to(device),
            input_hw=[data["input_hw"]],
            original_hw=[data["original_hw"]],
            intrinsics=data["intrinsics"].to(device)[None],
            padding=[data["padding"]],
            input_texts=[input_texts],
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    inference_s = t1 - t0

    # Save the prediction for visualization
    t0 = time.perf_counter()
    imshow_bboxes3d(
        image=data["original_images"].cpu(),
        boxes3d=[b.cpu() for b in boxes3d],
        intrinsics=data["original_intrinsics"].cpu().numpy(),
        scores=[s.cpu() for s in scores],
        class_ids=[c.cpu() for c in class_ids],
        class_id_mapping=class_id_mapping,
        file_path=str(output_image_path),
        n_colors=len(class_id_mapping),
    )
    _save_input_output_preview(
        input_image_path, output_image_path, compare_image_path
    )
    t1 = time.perf_counter()
    visualization_s = t1 - t0

    model_load_total_s = model_build_s + checkpoint_load_s

    print(f"[3D-MOOD demo] device={device}")
    print(f"[Timing] model_build (init + .to(device)): {model_build_s:.3f} s")
    print(f"[Timing] checkpoint_load:              {checkpoint_load_s:.3f} s")
    print(f"[Timing] model_load_total:             {model_load_total_s:.3f} s")
    print(f"[Timing] inference (forward only):     {inference_s:.3f} s")
    print(f"[Timing] visualization + compare png:  {visualization_s:.3f} s")
    print(f"[Paths] input:  {input_image_path}")
    print(f"[Paths] output: {output_image_path}")
    print(f"[Paths] compare (input|output): {compare_image_path}")
