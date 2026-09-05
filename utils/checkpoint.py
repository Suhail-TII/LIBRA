"""
utils/checkpoint.py
───────────────────
Instruction-level checkpointing for LIBRA experiments.

Each experiment saves results incrementally after every instruction.
On restart, completed instructions are detected and skipped automatically.

USAGE (in experiment runner inner loop):

    from utils.checkpoint import CheckpointManager

    ckpt = CheckpointManager(output_dir, model_name, exp_num)
    completed = ckpt.get_completed_indices()   # set of instruction_idx already done

    for idx, instruction in enumerate(instructions):
        if idx in completed:
            print(f"  Skipping instruction {idx} (already in checkpoint)")
            continue

        # ... run attack variants, collect rows ...

        ckpt.save(new_rows_df)   # append after each instruction
"""

import pandas as pd
from pathlib import Path
from typing import Set, Optional
import logging
import json
import time

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages incremental checkpointing for a single (experiment, model) run.

    Checkpoint files are saved as:
        <output_dir>/checkpoints/exp{N}_{model_name}_checkpoint.csv
        <output_dir>/checkpoints/exp{N}_{model_name}_checkpoint_meta.json

    On restart:
        - Loads existing checkpoint CSV
        - Returns set of already-completed instruction indices
        - New results are appended (not rewritten) for efficiency
    """

    def __init__(
        self,
        output_dir:  str | Path,
        model_name:  str,
        exp_num:     int,
    ):
        self.output_dir  = Path(output_dir)
        self.model_name  = model_name.replace("/", "_").replace(" ", "_")
        self.exp_num     = exp_num

        self.ckpt_dir    = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path    = self.ckpt_dir / f"exp{exp_num}_{self.model_name}_checkpoint.csv"
        self.meta_path   = self.ckpt_dir / f"exp{exp_num}_{self.model_name}_checkpoint_meta.json"

        self._existing_df: Optional[pd.DataFrame] = None
        self._header_written = self.csv_path.exists()

        logger.info(f"CheckpointManager: {self.csv_path}")

    def get_completed_indices(self) -> Set[int]:
        """
        Return set of instruction_idx values already saved in checkpoint.
        Empty set if no checkpoint exists yet.
        """
        if not self.csv_path.exists():
            logger.info("No checkpoint found — starting fresh")
            return set()

        try:
            df = pd.read_csv(self.csv_path)
            self._existing_df = df
            if 'instruction_idx' not in df.columns:
                logger.warning("Checkpoint CSV has no instruction_idx column — starting fresh")
                return set()
            completed = set(df['instruction_idx'].unique().tolist())
            logger.info(
                f"Checkpoint loaded: {len(completed)} instructions already done "
                f"({len(df)} rows) — resuming from next"
            )
            return completed
        except Exception as e:
            logger.warning(f"Could not read checkpoint: {e} — starting fresh")
            return set()

    def save(self, rows: pd.DataFrame):
        """
        Append a batch of result rows to the checkpoint CSV.
        Call this once per instruction (after all variants for that instruction complete).

        Args:
            rows: DataFrame with results for one instruction across all variants.
        """
        if rows.empty:
            return
        try:
            rows.to_csv(
                self.csv_path,
                mode='a',
                header=not self._header_written,
                index=False,
            )
            self._header_written = True
            self._update_meta(len(rows))
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_all(self) -> pd.DataFrame:
        """Load the full checkpoint CSV as a DataFrame."""
        if not self.csv_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.csv_path)

    def _update_meta(self, n_new_rows: int):
        """Update checkpoint metadata (last save time, row count)."""
        meta = {}
        if self.meta_path.exists():
            try:
                with open(self.meta_path) as f:
                    meta = json.load(f)
            except Exception:
                pass
        meta['last_saved']   = time.strftime("%Y-%m-%d %H:%M:%S")
        meta['total_rows']   = meta.get('total_rows', 0) + n_new_rows
        meta['model']        = self.model_name
        meta['exp_num']      = self.exp_num
        with open(self.meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def status(self) -> dict:
        """Return a summary dict of checkpoint status."""
        if not self.csv_path.exists():
            return {'exists': False, 'completed_instructions': 0, 'total_rows': 0}
        meta = {}
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                meta = json.load(f)
        completed = self.get_completed_indices()
        return {
            'exists':                  True,
            'completed_instructions':  len(completed),
            'total_rows':              meta.get('total_rows', 0),
            'last_saved':              meta.get('last_saved', 'unknown'),
            'checkpoint_path':         str(self.csv_path),
        }