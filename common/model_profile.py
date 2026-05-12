from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple
import time

import torch


@dataclass
class ModelProfile:
    input_shape: Tuple[int, ...]
    total_params: int
    trainable_params: int
    frozen_params: int
    flops: Optional[int]
    avg_inference_ms: float
    throughput_images_per_sec: float


def _format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _format_flops(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} GFLOPs"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} MFLOPs"
    if value >= 1_000:
        return f"{value / 1_000:.2f} KFLOPs"
    return f"{value} FLOPs"


def _profile_flops(model: torch.nn.Module, sample_input: torch.Tensor, device: torch.device) -> Optional[int]:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.inference_mode():
            with torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_flops=True,
            ) as prof:
                model(sample_input)

        total_flops = 0
        for event in prof.key_averages():
            event_flops = getattr(event, "flops", None)
            if event_flops:
                total_flops += int(event_flops)
        return total_flops if total_flops > 0 else None
    except Exception:
        return None


def profile_model(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: Sequence[int],
    warmup: Optional[int] = None,
    repeats: Optional[int] = None,
) -> ModelProfile:
    if warmup is None:
        warmup = 5 if device.type == "cuda" else 1
    if repeats is None:
        repeats = 20 if device.type == "cuda" else 3

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    param = next(model.parameters(), None)
    dtype = param.dtype if param is not None else torch.float32
    sample_input = torch.randn(*input_shape, device=device, dtype=dtype)

    was_training = model.training
    model.eval()

    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(warmup):
            model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(repeats):
            model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    if was_training:
        model.train()

    avg_inference_ms = (elapsed / max(repeats, 1)) * 1000.0
    throughput_images_per_sec = (input_shape[0] * repeats / elapsed) if elapsed > 0 else 0.0
    flops = _profile_flops(model, sample_input, device)

    return ModelProfile(
        input_shape=tuple(int(v) for v in input_shape),
        total_params=total_params,
        trainable_params=trainable_params,
        frozen_params=frozen_params,
        flops=flops,
        avg_inference_ms=avg_inference_ms,
        throughput_images_per_sec=throughput_images_per_sec,
    )


def model_profile_to_dict(profile: ModelProfile) -> Dict[str, object]:
    return asdict(profile)


def format_model_profile(profile: ModelProfile) -> str:
    return "\n".join(
        [
            f"Input shape: {profile.input_shape}",
            f"Total params: {_format_count(profile.total_params)}",
            f"Trainable params: {_format_count(profile.trainable_params)}",
            f"Frozen params: {_format_count(profile.frozen_params)}",
            f"FLOPs: {_format_flops(profile.flops)}",
            f"Inference time: {profile.avg_inference_ms:.2f} ms/image",
            f"Throughput: {profile.throughput_images_per_sec:.2f} images/s",
        ]
    )
