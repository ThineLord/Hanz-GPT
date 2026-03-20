from contextlib import nullcontext

import torch


def get_device_type(device: str) -> str:
    if "cuda" in device:
        return "cuda"
    if "mps" in device:
        return "mps"
    return "cpu"


def get_ptdtype(dtype: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]


def setup_torch_runtime(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def build_autocast_context(device: str, dtype: str):
    device_type = get_device_type(device)
    ptdtype = get_ptdtype(dtype)
    if device_type == "cpu":
        return device_type, ptdtype, nullcontext()
    return device_type, ptdtype, torch.amp.autocast(device_type=device_type, dtype=ptdtype)
