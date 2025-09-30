import os
import math
from typing import Dict, List, Optional

from collections import defaultdict

from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter


class BaseOutputFormat:
    """Abstract base class for output formats."""

    def writekvs(self, kvs: Dict[str, float], step: int) -> None:
        raise NotImplementedError


class ConsoleOutput(BaseOutputFormat):
    """Write key-value pairs as a formatted table to stdout."""

    def __init__(self, table_format: str = "simple") -> None:
        self.table_format = table_format

    def writekvs(self, kvs: Dict[str, float], step: int) -> None:  # noqa: D401
        if not kvs:
            return
        rows = [(k, f"{v:.6g}") for k, v in sorted(kvs.items())]
        print("\n===== Logger dump (step", step, ") =====")
        print(tabulate(rows, headers=["key", "value"], tablefmt=self.table_format))
        print("================================\n")


class TensorboardOutput(BaseOutputFormat):
    """Write scalars to TensorBoard."""

    def __init__(self, log_dir: str, writer: Optional[SummaryWriter] = None) -> None:
        self.writer = writer or SummaryWriter(log_dir=log_dir)

    def writekvs(self, kvs: Dict[str, float], step: int) -> None:
        for k, v in kvs.items():
            self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        self.writer.close()


class Logger:
    """Aggregates scalars and periodically dumps them to multiple outputs.

    This class is intentionally lightweight; it performs no disk I/O except via
    the chosen OutputFormats (console, tensorboard, etc.).
    """

    def __init__(
        self,
        log_dir: str,
        enabled_outputs: Optional[List[str]] = None,
        log_interval: int = 100,
        table_format: str = "simple",
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        self.log_interval = log_interval
        enabled_outputs = enabled_outputs or ["console"]

        self.output_formats: List[BaseOutputFormat] = []
        for out in enabled_outputs:
            if out == "console":
                self.output_formats.append(ConsoleOutput(table_format))
            elif out == "tensorboard":
                self.output_formats.append(TensorboardOutput(log_dir, writer))
            else:
                raise ValueError(f"Unknown logging output '{out}'")

        # mean accumulation buffers
        self.name2val: Dict[str, float] = defaultdict(float)
        self.name2cnt: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public accumulation methods
    # ------------------------------------------------------------------
    def logkv(self, key: str, val: float) -> None:
        """Log an instantaneous value (no mean aggregation)."""
        self.name2val[key] = val
        # ensure it is dumped even with mean logic; cnt=1 marks as non-mean
        self.name2cnt[key] = 1

    def logkv_mean(self, key: str, val: float) -> None:
        """Accumulate a value toward its running mean."""
        old = self.name2val.get(key, 0.0)
        cnt = self.name2cnt.get(key, 0)
        new_val = (old * cnt + float(val)) / (cnt + 1)
        self.name2val[key] = new_val
        self.name2cnt[key] = cnt + 1

    # ------------------------------------------------------------------
    # Diffusion-specific helper
    # ------------------------------------------------------------------
    def logkv_loss_quartiles(
        self,
        diffusion,  # has .num_timesteps
        ts,  # torch.Tensor[int] shape [batch]
        losses: Dict[str, "torch.Tensor"],  # key -> per-sample tensor same shape
    ) -> None:
        import torch  # local import to avoid torch in logger namespace for tools w/o torch

        num_steps = diffusion.num_timesteps
        if num_steps == 0:
            return

        for key, value_tensor in losses.items():
            if value_tensor.numel() == 0:
                continue
            # ensure 1-D same length as ts
            vals = value_tensor.detach().flatten()
            for sub_t, sub_loss in zip(ts.detach().cpu().tolist(), vals.cpu().tolist()):
                quartile = min(3, int(4 * sub_t / num_steps))
                self.logkv_mean(f"{key}_q{quartile}", sub_loss)

    # ------------------------------------------------------------------
    def dumpkvs(self, step: int) -> None:
        kvs = {k: float(v) for k, v in self.name2val.items()}
        for fmt in self.output_formats:
            fmt.writekvs(kvs, step)
        # reset means after dump
        self.name2val.clear()
        self.name2cnt.clear()

    def clear_accumulators(self) -> None:
        """Explicitly clear the accumulation buffers for manual resets."""
        self.name2val.clear()
        self.name2cnt.clear()

    def close(self) -> None:
        for fmt in self.output_formats:
            if isinstance(fmt, TensorboardOutput):
                fmt.close()