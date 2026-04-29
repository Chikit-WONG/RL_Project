"""General-purpose utilities: I/O, smoothing, summary statistics, and export."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from config import Config, ExperimentSpec


def smooth_curve(values: Sequence[float], window: int) -> np.ndarray:
    """Return a simple moving-average smoothed version of the input sequence."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    if array.size < window:
        return np.full_like(array, np.mean(array), dtype=np.float64)
    weights = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(array, weights, mode="valid")
    prefix = np.full(window - 1, smoothed[0], dtype=np.float64)
    return np.concatenate([prefix, smoothed])


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create and return the standard output directory layout."""

    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    logs_dir = output_dir / "logs"
    for directory in (output_dir, figures_dir, tables_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {"root": output_dir, "figures": figures_dir, "tables": tables_dir, "logs": logs_dir}


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write a list of dictionaries to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_name(text: str) -> str:
    """Convert a label to a filesystem-safe name."""

    return text.replace(" ", "_").replace("=", "").replace(".", "_")


def export_run_logs(result: Dict[str, object], logs_dir: Path) -> None:
    """Write per-run training and evaluation logs to disk."""

    spec: ExperimentSpec = result["spec"]
    stem = f"{spec.name}_seed{result['seed']}"
    write_csv(logs_dir / f"{stem}_train.csv", result["train_rows"])
    write_csv(logs_dir / f"{stem}_eval.csv", result["eval_rows"])


def filter_specs(all_specs: Sequence[ExperimentSpec], exp_choice: str) -> List[ExperimentSpec]:
    """Select the experiments requested by the CLI.

    ``exp_choice`` can be ``"all"`` or a prefix such as ``"exp1"``, ``"exp4"``,
    etc.  Exact name matches and ``<prefix>_*`` prefix matches are both included.
    """

    if exp_choice == "all":
        return list(all_specs)
    return [
        spec for spec in all_specs
        if spec.name == exp_choice or spec.name.startswith(exp_choice + "_")
    ]


def estimate_convergence_episode(eval_rows: Sequence[Dict[str, object]], threshold: float) -> Optional[int]:
    """Estimate the first evaluation point that reaches the target threshold."""

    for row in eval_rows:
        if float(row["success_rate"]) >= threshold:
            return int(row["episode"])
    return None


def build_run_summary(result: Dict[str, object], config: Config) -> Dict[str, object]:
    """Create a compact summary row for one training run."""

    spec: ExperimentSpec = result["spec"]
    eval_rows = result["eval_rows"]
    rewards = result["rewards"]
    smoothed_rewards = result["smoothed_rewards"]
    target_threshold = 0.95 if spec.name == "exp1" else 0.65
    convergence_episode = estimate_convergence_episode(eval_rows, target_threshold)
    final_success = float(eval_rows[-1]["success_rate"]) if eval_rows else math.nan
    best_success = float(max(row["success_rate"] for row in eval_rows)) if eval_rows else math.nan
    return {
        "condition": spec.name,
        "title": spec.title,
        "family": spec.family,
        "seed": result["seed"],
        "backend": result["backend_used"],
        "alpha": spec.alpha,
        "is_slippery": spec.is_slippery,
        "episodes": config.n_episodes,
        "final_eval_success": final_success,
        "best_eval_success": best_success,
        "final_smoothed_reward": float(smoothed_rewards[-1]) if smoothed_rewards.size else math.nan,
        "mean_reward": float(np.mean(rewards)) if rewards.size else math.nan,
        "convergence_episode": convergence_episode if convergence_episode is not None else "",
    }


def build_aggregate_summary(
    grouped_results: Mapping[str, Sequence[Dict[str, object]]],
    config: Config,
) -> List[Dict[str, object]]:
    """Aggregate seed-wise statistics for each condition."""

    rows: List[Dict[str, object]] = []
    for condition, runs in grouped_results.items():
        template_spec: ExperimentSpec = runs[0]["spec"]
        run_summaries = [build_run_summary(run, config) for run in runs]
        final_successes = np.asarray([row["final_eval_success"] for row in run_summaries], dtype=np.float64)
        best_successes = np.asarray([row["best_eval_success"] for row in run_summaries], dtype=np.float64)
        mean_rewards = np.asarray([row["mean_reward"] for row in run_summaries], dtype=np.float64)
        convergence_values = np.asarray(
            [row["convergence_episode"] for row in run_summaries if row["convergence_episode"] != ""],
            dtype=np.float64,
        )
        rows.append(
            {
                "condition": condition,
                "title": template_spec.title,
                "family": template_spec.family,
                "alpha": template_spec.alpha,
                "is_slippery": template_spec.is_slippery,
                "n_seeds": len(runs),
                "mean_final_eval_success": float(np.mean(final_successes)),
                "std_final_eval_success": float(np.std(final_successes)),
                "mean_best_eval_success": float(np.mean(best_successes)),
                "std_best_eval_success": float(np.std(best_successes)),
                "mean_reward": float(np.mean(mean_rewards)),
                "std_reward": float(np.std(mean_rewards)),
                "mean_convergence_episode": float(np.mean(convergence_values))
                if convergence_values.size
                else "",
                "std_convergence_episode": float(np.std(convergence_values))
                if convergence_values.size
                else "",
            }
        )
    return rows
