#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(
    eval_program_path="evaluate.py",
    extra_cmd_args={
        # Keep evaluation lightweight and deterministic.
        "warmup_iters": 2,
        "bench_iters": 4,
        "device": "cuda",
        "dtype": "auto",
    },
)

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
else:
    parent_config = dict(parent_selection_strategy="power_law")


db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
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

search_task_sys_msg = """You are a world-class GPU kernel engineer focused on Triton.
You must optimize three kernels used in LLM workloads:
1) fast_rmsnorm (RMSNorm forward/back)
2) apply_rope_triton (rotary embeddings, forward/back)
3) fast_relusqr (ReLU squared activation)

Constraints and goals:
- Preserve numerical correctness against PyTorch references on synthetic LLM-shaped tensors.
- Optimize for low forward/backward latency and high throughput; evaluation tracks speedups vs PyTorch.
- Favor FP16/BF16-friendly math while maintaining stability (epsilon handling, safe reductions).
- Keep autograd compatibility (no silent in-place issues), and avoid excessive memory use.
- Be explicit about grid/block choices, divergence avoidance, and vectorization.
- Target Hopper/Ampere-style GPUs but keep code portable across CUDA devices."""


evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=300,
    max_parallel_jobs=4,
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
    init_program_path="initial.py",
    results_dir="results_kernel_design",
)


def main():
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    main()
