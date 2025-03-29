import os
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    def __init__(
        self,
        log_dir: str,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        use_wandb: bool = True,
        use_tensorboard: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Logger supporting WandB, TensorBoard, and local logging.

        Args:
            log_dir (str): Local logging directory.
            project (str): WandB project name (if using WandB).
            run_name (str): Optional run name for WandB and file logs.
            use_wandb (bool): Whether to log to WandB.
            use_tensorboard (bool): Whether to use TensorBoard.
            config (dict): Optional configuration dictionary to log.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
        self.json_log_path = self.log_dir / f"{self.run_name}.jsonl"
        self.local_log_file = open(self.json_log_path, "a", buffering=1)

        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE

        if self.use_wandb:
            wandb.init(
                project=project,
                name=self.run_name,
                dir=str(self.log_dir),
                config=config,
                resume="allow",
                mode="online",
            )

        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=self.log_dir / "tensorboard")

        self._log = logging.getLogger("train_logger")
        self._log.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_dir / f"{self.run_name}.log")
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        self._log.addHandler(handler)

    def log(self, metrics: Dict[str, float], step: Optional[int] = None, split: str = "train"):
        """
        Log a dictionary of metrics.

        Args:
            metrics (dict): Key-value pairs of metric names and values.
            step (int): Global step or epoch number.
            split (str): Data split identifier ("train", "val", etc.)
        """
        step = step or metrics.get("step", 0)
        flat_metrics = {f"{split}/{k}": v for k, v in metrics.items() if isinstance(v, (float, int))}

        # Write to WandB
        if self.use_wandb:
            wandb.log(flat_metrics, step=step)

        # Write to TensorBoard
        if self.use_tensorboard:
            for k, v in flat_metrics.items():
                self.tb_writer.add_scalar(k, v, global_step=step)

        # Write to local file as JSON lines
        log_entry = {"step": step, "split": split, "metrics": flat_metrics}
        self.local_log_file.write(json.dumps(log_entry) + "\n")

        # Logging to terminal/log file
        msg = f"[{split}] Step {step} | " + " | ".join(f"{k}: {v:.4f}" for k, v in flat_metrics.items())
        self._log.info(msg)

    def close(self):
        """Finalize all logging outputs."""
        if self.use_wandb:
            wandb.finish()
        if self.use_tensorboard:
            self.tb_writer.close()
        self.local_log_file.close()
