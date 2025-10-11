#!/usr/bin/env python3
"""Launch evolutionary search for the Ariadne SDPA example."""

from pathlib import Path

from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

BASE_DIR = Path(__file__).resolve().parent

job_config = LocalJobConfig(eval_program_path=str(BASE_DIR / "evaluate.py"))

strategy = "weighted"
if strategy == "uniform":
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=0.0,
        exploitation_ratio=1.0,
    )
elif strategy == "hill_climbing":
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=100.0,
        exploitation_ratio=1.0,
    )
elif strategy == "weighted":
    parent_config = dict(
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )
elif strategy == "power_law":
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=1.0,
        exploitation_ratio=0.2,
    )
elif strategy == "power_law_high":
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=2.0,
        exploitation_ratio=0.2,
    )
elif strategy == "beam_search":
    parent_config = dict(
        parent_selection_strategy="beam_search",
        num_beams=10,
    )
else:
    raise ValueError(f"Unsupported strategy: {strategy}")

db_config = DatabaseConfig(
    db_path="evolution_ariadne_sdpa.sqlite",
    num_islands=2,
    archive_size=40,
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    migration_interval=10,
    migration_rate=0.1,
    island_elitism=True,
    **parent_config,
)

search_task_sys_msg = """You are optimizing a streaming scaled dot-product attention kernel.

Goals:
- Preserve exactness and numerical stability versus a high-precision SDPA reference.
- Reduce end-to-end latency and memory pressure, especially for fp8 K/V regimes.
- Surface accurate performance metadata (latency, bytes) to help the evaluator score improvements.

Consider:
- Tile sizes (BLOCK_M/N/D), warp and stage counts, and persistent row handling.
- Fusing fp8 scale application, handling per-row scales, and minimizing redundant conversion.
- Memory hierarchy usage (L1, shared memory) and accumulation stability (Kahan or better).
- Keeping the run_ariadne interface contract identical: return (outputs, reported_score, perf).
"""

evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=200,
    max_parallel_jobs=2,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        "o4-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        reasoning_efforts=["auto", "low", "medium", "high"],
        max_tokens=32768,
    ),
    meta_rec_interval=10,
    meta_llm_models=["gpt-5-nano"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    embedding_model="text-embedding-3-small",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=["gpt-5-nano"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    init_program_path=str(BASE_DIR / "initial.py"),
    results_dir=str(BASE_DIR / "results_ariadne_sdpa"),
)


def main() -> None:
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    main()
