"""Device selection and the float64 path the Apple GPU cannot hold."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.common import SelfAttention
from utils.device import accelerator, as_float64, visible_accelerators


def main():
    dev = accelerator()
    ids = visible_accelerators("")
    assert ids, "expected a CUDA device or the Apple GPU"
    if dev.type == "mps":
        assert ids == ["mps"]
        t = torch.randn(4, 8, device=dev)
        try:
            t.double()
            raise SystemExit("Metal accepted float64")
        except TypeError:
            pass
        out = as_float64(t)
        assert out.dtype == torch.float64 and out.device.type == "cpu"
        assert torch.allclose(out, t.detach().float().cpu().double())
    elif dev.type == "cuda":
        assert ids == [str(i) for i in range(torch.cuda.device_count())]
        t = torch.randn(4, 8, device=dev)
        out = as_float64(t)
        assert out.dtype == torch.float64 and out.device.type == "cuda"

    if dev.type != "cpu":
        cfg = SimpleNamespace(n_embd=32, n_head=4, bias=True, dropout=0.0)
        attn = SelfAttention(cfg).to(dev).eval()
        x = torch.randn(2, 8, 32, device=dev)
        mask = torch.zeros(1, 1, 8, 8, device=dev)
        with torch.no_grad():
            y = attn(x, attn_mask=mask)
        assert y.shape == x.shape and y.device.type == dev.type
        assert torch.isfinite(y).all()
    print(f"ok {dev}")


if __name__ == "__main__":
    main()
