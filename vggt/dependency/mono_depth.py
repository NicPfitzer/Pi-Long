import logging
import os
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _load_midas(device: torch.device):
    """Load MiDaS via torch.hub as a robust fallback monocular depth model.

    Returns a tuple of (model, transform_fn_name) where transform_fn_name selects
    the correct preprocessing transform from MiDaS' transforms module.
    """
    try:
        model_type = "DPT_BEiT_L_512"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.eval().to(device)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = transforms.dpt_beit512
        return midas, transform
    except Exception as e:
        logger.error("Failed to load MiDaS via torch.hub: %s", e)
        raise


def _infer_midas(
    image_paths: List[str],
    device: torch.device,
    resize_to_input: bool = True,
    dtype: torch.dtype = torch.float32,
) -> List[np.ndarray]:
    """Run MiDaS on a list of image paths and return depths resized to original image sizes.

    The returned depths are arbitrary-scale and shifted (relative), suitable for
    subsequent scale/shift alignment using SfM points.
    """
    model, transform = _load_midas(device)
    results: List[np.ndarray] = []
    for p in image_paths:
        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        input_batch = transform(img_rgb).to(device=device, dtype=dtype).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_batch)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1)
            depth = pred[0].detach().float().cpu().numpy()
        results.append(depth.astype(np.float32))
    return results


_MOGE2_CACHED = None  # (model, default_pretrained)
_DA2_CACHE: Dict[tuple, torch.nn.Module] = {}

_DEPTH_ANYTHING_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
}

_DEPTH_ANYTHING_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def _try_load_moge2(device: torch.device, pretrained: Optional[str] = None):
    """Best-effort import for MoGe v2.

    Returns a callable taking a list of image paths and returning list of depth maps.
    If MoGe v2 is not available, returns None.
    """
    try:
        # pip install git+https://github.com/microsoft/MoGe.git
        from moge.model.v2 import MoGeModel  # type: ignore

        global _MOGE2_CACHED
        if _MOGE2_CACHED is None or (pretrained and _MOGE2_CACHED[1] != pretrained):
            name = pretrained or "Ruicheng/moge-2-vitl-normal"
            model = MoGeModel.from_pretrained(name).to(device).eval()
            _MOGE2_CACHED = (model, name)
        else:
            model = _MOGE2_CACHED[0]

        def run_moge2(paths: List[str]) -> List[np.ndarray]:
            outs: List[np.ndarray] = []
            for p in paths:
                img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise FileNotFoundError(p)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                ten = torch.from_numpy(img_rgb).to(device).float() / 255.0
                ten = ten.permute(2, 0, 1)  # (3,H,W)
                with torch.no_grad():
                    out = model.infer(ten)
                depth = out.get("depth")
                if depth is None:
                    raise RuntimeError("MoGe-2 infer() did not return 'depth'")
                depth_np = depth.detach().float().cpu().numpy()
                outs.append(depth_np.astype(np.float32))
            return outs

        return run_moge2
    except Exception as e:
        logger.warning("MoGe v2 not available or failed to import: %s", e)
        return None


def _try_load_depthanything_v2(
    device: torch.device,
    *,
    encoder: str = "vitl",
    checkpoint: Optional[str] = None,
    input_size: int = 518,
) -> Optional[Callable[[List[str]], List[np.ndarray]]]:
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except Exception as e:
        logger.warning("Depth Anything V2 not available or failed to import: %s", e)
        return None

    encoder = encoder.lower()
    if encoder not in _DEPTH_ANYTHING_CONFIGS:
        logger.error("Unsupported Depth Anything V2 encoder '%s'.", encoder)
        return None

    cache_id = checkpoint or _DEPTH_ANYTHING_URLS.get(encoder, "")
    cache_key = (encoder, cache_id, str(device))

    model = _DA2_CACHE.get(cache_key)
    if model is None:
        cfg = _DEPTH_ANYTHING_CONFIGS[encoder]
        model = DepthAnythingV2(**cfg)

        state_dict = None
        if checkpoint is not None:
            if os.path.isfile(checkpoint):
                state_dict = torch.load(checkpoint, map_location="cpu")
            elif checkpoint.startswith("http://") or checkpoint.startswith("https://"):
                state_dict = torch.hub.load_state_dict_from_url(checkpoint, map_location="cpu")
            else:
                logger.info(
                    "Depth Anything V2 checkpoint '%s' not recognized as file or URL; using default weights for %s.",
                    checkpoint,
                    encoder,
                )

        if state_dict is None:
            url = _DEPTH_ANYTHING_URLS.get(encoder)
            if url is None:
                raise RuntimeError(f"No default checkpoint URL for encoder '{encoder}'.")
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")

        model.load_state_dict(state_dict)
        model = model.to(device=device).eval()
        _DA2_CACHE[cache_key] = model
    else:
        model = model.to(device=device).eval()

    def run_da2(paths: List[str]) -> List[np.ndarray]:
        outs: List[np.ndarray] = []
        for p in paths:
            img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(p)
            depth = model.infer_image(img_bgr, input_size)
            outs.append(depth.astype(np.float32))
        return outs

    return run_da2


def predict_monocular_depths(
    image_paths: List[str],
    model_name: str,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
    pretrained: Optional[str] = None,
    model_kwargs: Optional[Dict[str, object]] = None,
) -> List[np.ndarray]:
    """Predict monocular depths for a sequence of images.

    - model_name: 'moge2', 'depthanythingv2', 'midas', or 'auto'.
    - Returns a list of HxW float32 depth maps per image, at original image size.
    """
    model_name = (model_name or "auto").lower()
    model_kwargs = model_kwargs or {}

    if model_name == "moge2" or model_name == "auto":
        runner: Optional[Callable[[List[str]], List[np.ndarray]]] = _try_load_moge2(device, pretrained=pretrained)
        if runner is not None:
            logger.info("Using MoGe v2 for monocular depth.")
            return runner(image_paths)
        if model_name == "moge2":
            logger.warning("Requested MoGe v2 but it is not available; falling back to MiDaS.")

    if model_name in {"depthanythingv2", "depthanything", "dav2", "auto"} or model_kwargs.get("encoder"):
        encoder = str(model_kwargs.get("encoder", "vitl"))
        input_size = int(model_kwargs.get("input_size", 518))
        runner = _try_load_depthanything_v2(
            device,
            encoder=encoder,
            checkpoint=pretrained,
            input_size=input_size,
        )
        if runner is not None:
            logger.info("Using Depth-Anything V2 (%s) for monocular depth.", encoder)
            return runner(image_paths)
        if model_name not in {"auto"}:
            logger.warning("Requested Depth-Anything V2 but it is not available; falling back to MiDaS.")

    logger.info("Using MiDaS fallback for monocular depth.")
    return _infer_midas(image_paths, device=device, dtype=dtype)
