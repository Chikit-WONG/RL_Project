"""Designed random-map experiments with explicit hole-count and solvability checks."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import enhanced_frozenlake as base
from map_env_utils import CustomMapFrozenLakeEnv, evaluate_on_env, make_random_map


ALGORITHMS = [
    "q_learning",
    "sarsa",
    "expected_sarsa",
    "sarsa_lambda",
    "q_lambda",
    "optimistic_q",
    "dyna_q",
]

DISPLAY = {
    "q_learning": "Q-learning",
    "sarsa": "SARSA",
    "expected_sarsa": "Expected SARSA",
    "sarsa_lambda": "SARSA(lambda)",
    "q_lambda": "Q(lambda)",
    "optimistic_q": "Optimistic Q",
    "dyna_q": "Dyna-Q",
}

STANDARD_4X4 = ["SFFF", "FHFH", "FFFH", "HFFG"]
STANDARD_8X8 = base.STANDARD_8X8


def hole_count(layout: Sequence[str]) -> int:
    return sum(row.count("H") for row in layout)


def shortest_path_length(layout: Sequence[str]) -> int | None:
    n = len(layout)
    start = (0, 0)
    goal = (n - 1, n - 1)
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            if (nr, nc) in seen or layout[nr][nc] == "H":
                continue
            seen.add((nr, nc))
            queue.append(((nr, nc), dist + 1))
    return None


def make_exact_hole_map(size: int, n_holes: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    cells = [(r, c) for r in range(size) for c in range(size) if (r, c) not in [(0, 0), (size - 1, size - 1)]]
    chosen = set(rng.choice(len(cells), size=n_holes, replace=False).tolist())
    grid = [["F" for _ in range(size)] for _ in range(size)]
    for idx, (r, c) in enumerate(cells):
        if idx in chosen:
            grid[r][c] = "H"
    grid[0][0] = "S"
    grid[-1][-1] = "G"
    return ["".join(row) for row in grid]


def oracle_score(layout: Sequence[str], eval_episodes: int, max_steps: int, seed: int) -> float:
    cfg = base.ExperimentConfig(eval_episodes=eval_episodes, max_steps=max_steps)
    env = CustomMapFrozenLakeEnv(layout, True, seed)
    oracle = base.ValueIterationAgent(env.n_states, env.n_actions, cfg, seed, env)
    return evaluate_on_env(oracle, lambda s: CustomMapFrozenLakeEnv(layout, True, s), cfg, seed + 777_000)


def select_designed_maps(
    size: int,
    n_holes: int,
    n_maps: int,
    min_oracle: float,
    max_oracle: float,
    eval_episodes: int,
    max_steps: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    attempts = 0
    candidate = 0
    while len(rows) < n_maps and attempts < 10_000:
        attempts += 1
        map_seed = seed + attempts
        layout = make_exact_hole_map(size, n_holes, map_seed)
        shortest = shortest_path_length(layout)
        if shortest is None:
            continue
        score = oracle_score(layout, eval_episodes, max_steps, map_seed)
        if score < min_oracle or score > max_oracle:
            continue
        rows.append(
            {
                "map_id": candidate,
                "map_seed": map_seed,
                "size": size,
                "holes": hole_count(layout),
                "shortest_path": shortest,
                "oracle_success": score,
                "layout": "|".join(layout),
            }
        )
        candidate += 1
    if len(rows) < n_maps:
        raise RuntimeError(f"Only found {len(rows)} designed maps after {attempts} attempts.")
    return pd.DataFrame(rows)


def train_agent(algorithm: str, env, cfg: base.ExperimentConfig, seed: int):
    agent = base.AGENTS[algorithm](env.n_states, env.n_actions, cfg, seed)
    for _ in range(cfg.episodes):
        agent.train_episode(env)
    return agent


def run_designed_4x4(args: argparse.Namespace, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)

    maps = select_designed_maps(
        size=4,
        n_holes=4,
        n_maps=args.maps,
        min_oracle=args.min_oracle,
        max_oracle=args.max_oracle,
        eval_episodes=args.oracle_eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    maps.to_csv(out / "tables" / "designed_4x4_map_manifest.csv", index=False)

    cfg = base.ExperimentConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        eval_interval=args.episodes,
    )
    cfg.planning_steps = args.planning_steps
    cfg.optimistic_init = args.optimistic_init

    algorithms = [item.strip() for item in args.algorithms.split(",") if item.strip()]
    rows = []
    for _, m in maps.iterrows():
        layout = str(m["layout"]).split("|")
        for algorithm in algorithms:
            for seed in range(args.seeds):
                env = CustomMapFrozenLakeEnv(layout, True, seed)
                agent = train_agent(algorithm, env, cfg, seed)
                score = evaluate_on_env(agent, lambda s, layout=layout: CustomMapFrozenLakeEnv(layout, True, s), cfg, seed + 900_000)
                rows.append(
                    {
                        "suite": "designed_4x4_exact4holes",
                        "algorithm": algorithm,
                        "display_algorithm": DISPLAY.get(algorithm, algorithm),
                        "env": "4x4_designed_stoch_exact4holes",
                        "map_id": int(m["map_id"]),
                        "seed": seed,
                        "success": score,
                        "oracle_success": float(m["oracle_success"]),
                        "holes": int(m["holes"]),
                        "shortest_path": int(m["shortest_path"]),
                        "layout": m["layout"],
                    }
                )
                pd.DataFrame(rows).to_csv(out / "tables" / "designed_4x4_runs_partial.csv", index=False)
                print("designed4x4", m["map_id"], algorithm, seed, score, "oracle", m["oracle_success"])

    runs = pd.DataFrame(rows)
    aggregate = (
        runs.groupby(["algorithm", "display_algorithm", "env"], as_index=False)
        .agg(
            mean_success=("success", "mean"),
            std_success=("success", "std"),
            n=("success", "count"),
            mean_oracle=("oracle_success", "mean"),
        )
        .sort_values("mean_success", ascending=False)
    )
    runs.to_csv(out / "tables" / "designed_4x4_runs.csv", index=False)
    aggregate.to_csv(out / "tables" / "designed_4x4_aggregate_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.barplot(
        data=runs,
        x="map_id",
        y="success",
        hue="display_algorithm",
        errorbar="sd",
        ax=ax,
    )
    oracle = maps[["map_id", "oracle_success"]].copy()
    ax.plot(
        oracle["map_id"],
        oracle["oracle_success"],
        color="black",
        linestyle="--",
        marker="D",
        linewidth=2.0,
        label="Value Iteration oracle",
    )
    ax.set_title("Designed 4x4 Slippery Random Maps: Exact 4 Holes")
    ax.set_xlabel("Designed map id")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figures" / "designed_4x4_exact4holes_success_by_map.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.barplot(data=aggregate, x="display_algorithm", y="mean_success", errorbar=None, ax=ax)
    ax.axhline(float(maps["oracle_success"].mean()), color="black", linestyle="--", linewidth=2.0, label="Mean oracle")
    ax.set_title("Designed 4x4 Exact-4-Hole Maps: Aggregate")
    ax.set_xlabel("")
    ax.set_ylabel("Mean success rate")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figures" / "designed_4x4_exact4holes_aggregate.png", dpi=220)
    plt.close(fig)


def audit_maps(out: Path, designed_dir: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for name, layout, source in [
        ("standard_4x4", STANDARD_4X4, "Gymnasium predefined"),
        ("standard_8x8", STANDARD_8X8, "Gymnasium predefined"),
    ]:
        rows.append(
            {
                "map_name": name,
                "source": source,
                "size": len(layout),
                "holes": hole_count(layout),
                "hole_ratio": hole_count(layout) / (len(layout) * len(layout)),
                "shortest_path": shortest_path_length(layout),
                "layout": "|".join(layout),
            }
        )
    if designed_dir is not None and (designed_dir / "tables" / "designed_4x4_map_manifest.csv").exists():
        manifest = pd.read_csv(designed_dir / "tables" / "designed_4x4_map_manifest.csv")
        for _, row in manifest.iterrows():
            rows.append(
                {
                    "map_name": f"designed_4x4_map_{int(row['map_id'])}",
                    "source": "Exact 4-hole designed random",
                    "size": int(row["size"]),
                    "holes": int(row["holes"]),
                    "hole_ratio": int(row["holes"]) / (int(row["size"]) ** 2),
                    "shortest_path": int(row["shortest_path"]),
                    "layout": row["layout"],
                }
            )
    for p in [0.80, 0.85, 0.90, 0.95]:
        for map_id in range(5):
            map_seed = 10_000 + 8 * 1000 + int(p * 100) * 10 + map_id
            layout = make_random_map(8, p, map_seed)
            rows.append(
                {
                    "map_name": f"current_8x8_random_p{p:.2f}_map_{map_id}",
                    "source": "Current carved-path random generator",
                    "size": 8,
                    "holes": hole_count(layout),
                    "hole_ratio": hole_count(layout) / 64,
                    "shortest_path": shortest_path_length(layout),
                    "layout": "|".join(layout),
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "map_design_audit.csv", index=False)

    markdown = [
        "# Map Design Audit",
        "",
        "Official Gymnasium documentation uses predefined 4x4 and 8x8 maps, and `is_slippery=True` means stochastic perpendicular slips. Random maps are useful, but they must be filtered so they are solvable and not trivial/extreme.",
        "",
        "## Audit Table",
        "",
        audit[["map_name", "source", "size", "holes", "hole_ratio", "shortest_path"]].to_markdown(index=False),
        "",
        "## Design Decisions",
        "",
        "- Standard 4x4 is kept as the main 4x4 benchmark: `SFFF/FHFH/FFFH/HFFG` with 4 holes (25%).",
        "- Designed 4x4 random maps use exactly 4 holes, BFS solvability, and Value Iteration oracle filtering.",
        "- Standard 8x8 is kept as the main fixed 8x8 benchmark.",
        "- Existing 8x8 random maps are slippery and useful for generalization, but their carved-path generator should be treated as a first generalization pass.",
        "",
        "## Layouts",
        "",
    ]
    for _, row in audit.iterrows():
        markdown.append(f"### {row['map_name']}")
        markdown.append("")
        markdown.append("```text")
        markdown.extend(str(row["layout"]).split("|"))
        markdown.append("```")
        markdown.append("")
    (out / "map_design_audit.md").write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("designed_map_results_v1"))
    parser.add_argument("--suites", default="designed4,audit")
    parser.add_argument("--maps", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--eval-episodes", type=int, default=150)
    parser.add_argument("--oracle-eval-episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--min-oracle", type=float, default=0.55)
    parser.add_argument("--max-oracle", type=float, default=0.95)
    parser.add_argument("--planning-steps", type=int, default=3)
    parser.add_argument("--optimistic-init", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=24_000)
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    args = parser.parse_args()

    suites = {item.strip() for item in args.suites.split(",") if item.strip()}
    if "designed4" in suites:
        run_designed_4x4(args, args.output_dir)
    if "audit" in suites:
        audit_maps(args.output_dir / "audit", args.output_dir)


if __name__ == "__main__":
    main()
