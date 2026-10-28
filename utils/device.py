import torch


def accelerator() -> torch.device:
    """CUDA when it is present, otherwise the Apple GPU, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def visible_accelerators(explicit: str = "") -> list[str]:
    """One entry per generation process: CUDA ids, or a single Apple GPU."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        if explicit:
            return explicit.split(",")
        return [str(i) for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        return ["mps"]
    return []


def as_float64(t: torch.Tensor) -> torch.Tensor:
    """Float64 values. Metal has no float64, so those tensors stay on the CPU."""
    if t.device.type == "mps":
        t = t.detach().to("cpu")
    return t.double()
