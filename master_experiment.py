"""
Master Experiment Runner V2

Runs all experiments across all models with:
- L1B3RT4S attack integration
- Three-tier ensemble evaluator (Heuristic + Llama Guard 3 + HarmBench)
- Per-model individual figures
- Aggregate cross-model visualizations
- AirLLM support for large models
- Proper GPU memory management

USAGE:
    # Run all experiments on AdvBench
    python master_experiment.py --dataset advbench

    # Run specific experiments with sample
    python master_experiment.py --dataset advbench --sample-size 50 --experiments 1,2,4

    # Run with L1B3RT4S attacks
    python master_experiment.py --dataset advbench --include-l1b3rt4s

    # Disable heavy classifiers for fast iteration (heuristic only)
    python master_experiment.py --dataset advbench --no-llama-guard --no-harmbench

    # Force classifiers to CPU (if VRAM is tight)
    python master_experiment.py --dataset advbench --evaluator-device cpu

    # Run Exp5 LIBRA (requires Exp4 results for baseline comparison)
    python master_experiment.py --dataset advbench --sample-size 50 --experiments 5

    # Run Exp5 pilot (10 instructions, quick sanity check)
    python master_experiment.py --dataset advbench --experiments 5 --exp5-pilot-n 10

    # Run Exp5 with all 12 UTOS templates (full scale: 57,600 combo tests)
    python master_experiment.py --dataset advbench --experiments 5 --exp5-utos all12

    # Run full pipeline Exp1-5
    python master_experiment.py --dataset advbench --sample-size 50 --experiments 1,2,3,4,5 --include-l1b3rt4s
"""

import sys
sys.path.append('.')

from dotenv import load_dotenv
load_dotenv()

import os
import gc
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import torch
import traceback


def load_models_config(config_path: str = "config/models.yaml") -> List[Dict]:
    """Load models from YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['models']


def load_dataset(dataset_name: str, sample_size: Optional[int] = None) -> List[str]:
    """Load harmful instructions dataset from HuggingFace."""
    from datasets import load_dataset as hf_load_dataset

    datasets_info = {
        'advbench': {
            'hf_name': 'walledai/AdvBench',
            'split': 'train',
            'column': 'prompt',
        },
        'harmbench': {
            'hf_name': 'walledai/HarmBench',
            'split': 'train',
            'column': 'prompt',
        },
        'malicious_instruct': {
            'hf_name': 'walledai/MaliciousInstruct',
            'split': 'train',
            'column': 'prompt',
        },
    }

    if dataset_name not in datasets_info:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    info = datasets_info[dataset_name]
    print(f"Loading {dataset_name} from HuggingFace...")
    print(f"  Dataset: {info['hf_name']}")

    dataset = hf_load_dataset(info['hf_name'])
    instructions = list(dataset[info['split']][info['column']])
    print(f"  Total instructions: {len(instructions)}")

    if sample_size and sample_size < len(instructions):
        import random
        random.seed(42)
        instructions = random.sample(instructions, sample_size)
        print(f"  Sampled: {sample_size} instructions")

    return instructions


def get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0
    try:
        total    = torch.cuda.get_device_properties(0).total_memory / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        return total - reserved
    except Exception:
        return 0


def cleanup_gpu():
    """Aggressive GPU cleanup between model loads."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


def load_model_from_config(model_config: Dict, force_eager_attention: bool = False):
    """Load a model based on YAML config with automatic strategy selection."""
    from models.model_wrapper import (
        load_model, get_gpu_memory_gb, determine_loading_strategy
    )

    provider = model_config['provider']
    api_name = model_config['api_name']
    print(f"  Loading {model_config['name']}...")

    if provider == 'HuggingFace':
        available_gb = get_gpu_memory_gb()
        strategy     = determine_loading_strategy(api_name, available_gb)
        print(f"    Strategy: {strategy['method']} ({strategy['reason']})")

        if strategy['method'] == 'airllm':
            model = load_model(
                'huggingface', api_name,
                use_airllm=True,
                airllm_compression=strategy['quantization'] or '4bit',
            )
        else:
            load_kwargs = {'force_eager_attention': force_eager_attention}
            if strategy['quantization'] == '8bit':
                load_kwargs['load_in_8bit'] = True
            elif strategy['quantization'] == '4bit':
                load_kwargs['load_in_4bit'] = True
            model = load_model('huggingface', api_name, **load_kwargs)
    else:
        provider_map = {
            'openai':    'gpt',
            'anthropic': 'claude',
            'google':    'gemini',
            'xai':       'xai',
        }
        model = load_model(provider_map.get(provider.lower(), 'gpt'), api_name)

    print(f"  ✓ Model loaded")
    return model


def _ensemble_col(df: pd.DataFrame) -> str:
    """Return ensemble column if present, else fall back to jailbreak_success."""
    if "jailbreak_success_ensemble" in df.columns:
        return "jailbreak_success_ensemble"
    return "jailbreak_success"


def _collect_asr_reports(exp_output_dir: Path) -> List[Dict]:
    """
    Scan an experiment output directory for ASR report JSONs and return their contents.
    Each experiment saves one JSON per model: exp{N}_asr_report_{model}_{ts}.json
    """
    reports = []
    for path in sorted(exp_output_dir.rglob("*asr_report*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            data['_source'] = str(path)
            reports.append(data)
        except Exception:
            pass
    return reports


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_1(
    models:           List[Dict],
    dataset:          List[str],
    output_dir:       Path,
    config:           Dict,
    include_l1b3rt4s: bool = False,
    evaluator_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """Run Experiment 1: Tokens vs Structure."""
    from experiments.exp1_tokens_vs_structure_v2 import (
        run_experiment_1_v2, create_aggregate_visualizations,
    )

    evaluator_kwargs = evaluator_kwargs or {}
    print(f"\n{'='*80}")
    print("EXPERIMENT 1: Tokens vs Structure")
    print(f"{'='*80}")

    all_dfs = []
    for model_config in models:
        print(f"\n{'─'*60}")
        print(f"Model: {model_config['name']}")
        print(f"{'─'*60}")
        try:
            cleanup_gpu()
            model   = load_model_from_config(model_config)
            results = run_experiment_1_v2(
                model=model,
                model_name=model_config['name'],
                harmful_instructions=dataset,
                config=config,
                include_l1b3rt4s=include_l1b3rt4s,
                **evaluator_kwargs,
            )
            if results and 'dataframe' in results:
                all_dfs.append(results['dataframe'])
                print(f"  ✓ Completed: {len(results['dataframe'])} tests")
            del model
            cleanup_gpu()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            traceback.print_exc()

    if all_dfs:
        combined_df   = pd.concat(all_dfs, ignore_index=True)
        combined_file = output_dir / "exp1_combined_results.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\n✓ Combined results: {combined_file}")
        print("\nCreating aggregate visualizations...")
        create_aggregate_visualizations(combined_df, output_dir)
        return combined_df
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_2(
    models:           List[Dict],
    dataset:          List[str],
    output_dir:       Path,
    config:           Dict,
    include_l1b3rt4s: bool = False,
    evaluator_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """Run Experiment 2: Lexical vs Structural."""
    from experiments.exp2_lexical_vs_structural_v2 import (
        run_experiment_2_v2, create_aggregate_visualizations,
    )

    evaluator_kwargs = evaluator_kwargs or {}
    print(f"\n{'='*80}")
    print("EXPERIMENT 2: Lexical vs Structural")
    print(f"{'='*80}")

    all_dfs = []
    for model_config in models:
        print(f"\n{'─'*60}")
        print(f"Model: {model_config['name']}")
        print(f"{'─'*60}")
        try:
            cleanup_gpu()
            model   = load_model_from_config(model_config)
            results = run_experiment_2_v2(
                model=model,
                model_name=model_config['name'],
                harmful_instructions=dataset,
                config=config,
                include_l1b3rt4s=include_l1b3rt4s,
                **evaluator_kwargs,
            )
            if results and 'dataframe' in results:
                all_dfs.append(results['dataframe'])
                print(f"  ✓ Completed: {len(results['dataframe'])} tests")
            del model
            cleanup_gpu()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            traceback.print_exc()

    if all_dfs:
        combined_df   = pd.concat(all_dfs, ignore_index=True)
        combined_file = output_dir / "exp2_combined_results.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\n✓ Combined results: {combined_file}")
        print("\nCreating aggregate visualizations...")
        create_aggregate_visualizations(combined_df, output_dir)
        return combined_df
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_3(
    models:           List[Dict],
    dataset:          List[str],
    output_dir:       Path,
    config:           Dict,
    evaluator_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """Run Experiment 3: Lexical x Structural Cross-Product Combination Analysis."""
    from experiments.exp3_lexical_structural_combination import (
        run_experiment_3, create_aggregate_exp3_visualizations,
    )

    evaluator_kwargs = evaluator_kwargs or {}
    print(f"\n{'='*80}")
    print("EXPERIMENT 3: Lexical x Structural Cross-Product")
    print(f"{'='*80}")

    all_dfs     = []
    all_synergy = []
    for model_config in models:
        print(f"\n{'─'*60}")
        print(f"Model: {model_config['name']}")
        print(f"{'─'*60}")
        try:
            cleanup_gpu()
            model   = load_model_from_config(model_config)
            results = run_experiment_3(
                model=model,
                model_name=model_config['name'],
                harmful_instructions=dataset,
                config=config,
                n_per_condition=50,   # pilot; change to 50 for full run
                **evaluator_kwargs,
            )
            if results and 'dataframe' in results:
                df          = results['dataframe']
                df['model'] = model_config['name']
                all_dfs.append(df)
                syn           = results['synergy_matrix']
                syn['model']  = model_config['name']
                all_synergy.append(syn)
                print(f"  ✓ Completed: {len(df)} tests")
            del model
            cleanup_gpu()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            traceback.print_exc()

    if all_dfs:
        combined_df      = pd.concat(all_dfs,     ignore_index=True)
        combined_synergy = pd.concat(all_synergy, ignore_index=True)
        combined_file    = output_dir / "exp3_combined_results.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\n✓ Combined results: {combined_file}")
        print("\nCreating aggregate visualizations...")
        create_aggregate_exp3_visualizations(combined_df, combined_synergy, output_dir)
        return combined_df
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 4
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_4(
    models:           List[Dict],
    dataset:          List[str],
    output_dir:       Path,
    config:           Dict,
    evaluator_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """Run Experiment 4: StructuralSleight Replication (Li et al., 2025)."""
    from experiments.exp4_structuralsleight_replication import (
        run_experiment_4 as run_ss,
        create_aggregate_exp4_visualizations,
    )
    import openai

    evaluator_kwargs = evaluator_kwargs or {}
    print(f"\n{'='*80}")
    print("EXPERIMENT 4: StructuralSleight Replication")
    print(f"{'='*80}")

    try:
        openai_client = openai.OpenAI()
        use_gpt4o     = True
        print("  ✓ GPT-4o dual evaluator enabled")
    except Exception:
        openai_client = None
        use_gpt4o     = False
        print("  ⚠ GPT-4o evaluator disabled (no API key)")

    all_dfs = []
    for model_config in models:
        print(f"\n{'─'*60}")
        print(f"Model: {model_config['name']}")
        print(f"{'─'*60}")
        try:
            cleanup_gpu()
            model   = load_model_from_config(model_config)
            results = run_ss(
                model=model,
                model_name=model_config['name'],
                harmful_instructions=dataset,
                config=config,
                use_gpt4o_evaluator=use_gpt4o,
                openai_client=openai_client,
                n_samples=50,  # pilot; change to 50 for full run
                **evaluator_kwargs,
            )
            if results and 'dataframe' in results:
                df          = results['dataframe']
                df['model'] = model_config['name']
                all_dfs.append(df)
                print(f"  ✓ Completed: {len(df)} tests")
            del model
            cleanup_gpu()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            traceback.print_exc()

    if all_dfs:
        combined_df   = pd.concat(all_dfs, ignore_index=True)
        combined_file = output_dir / "exp4_combined_results.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\n✓ Combined results: {combined_file}")
        print("\nCreating aggregate visualizations...")
        create_aggregate_exp4_visualizations(combined_df, output_dir)
        return combined_df
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 5  — LIBRA (L1B3RT4S x UTOS)
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_5(
    models:           List[Dict],
    dataset:          List[str],
    output_dir:       Path,
    config:           Dict,
    utos_templates:   str = "top4",
    pilot_n:          int = 0,
    exp4_csv:         Optional[str] = None,
    evaluator_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Run Experiment 5: LIBRA Evaluation
    ===================================
    8 L1B3RT4S variants (LIBRA) x UTOS templates x 2 composition orders x 6 models.

    LIBRA variants    : 8 L1B3RT4S attacks from Exp1
    UTOS templates    : top-4 from Exp4 SA greedy by default (tree, sequence_diagram,
                        markdown_table, json); pass utos_templates='all12' for full scale
    Composition orders: libra_then_utos (LIBRA outer) and utos_then_libra (UTOS outer)
    Controls          : LIBRA-only and UTOS-only standalone runs for synergy scoring
    Baseline          : Exp4 SS-SA/SCA/FSA results loaded from CSV (not re-run)

    Scale (top-4, 6 models, 50 instructions):
      Combinations : 8 x 4 x 2 x 50 x 6 = 19,200
      Controls     : (8 + 4) x 50 x 6   =  3,600
      Total        :                       22,800 tests

    Args:
        utos_templates   : "top4" | "all12" | comma-separated list of template names
        pilot_n          : if > 0, run only first pilot_n instructions per model
        exp4_csv         : explicit path to Exp4 combined CSV for baseline comparison;
                           if None, runner searches automatically
        evaluator_kwargs : dict forwarded to run_experiment_5() (llama_guard, harmbench etc.)
    """
    from experiments.exp5_libra_utos import (
        run_experiment_5 as _run_exp5,
        compare_libra_vs_structuralsleight,
        create_aggregate_exp5_visualizations,
    )
    import openai

    evaluator_kwargs = evaluator_kwargs or {}
    print(f"\n{'='*80}")
    print("EXPERIMENT 5: LIBRA Evaluation (L1B3RT4S x UTOS)")
    print(f"{'='*80}")
    print(f"  LIBRA variants    : 8 (all L1B3RT4S attacks from Exp1)")
    print(f"  UTOS templates    : {utos_templates}")
    print(f"  Composition orders: libra_then_utos, utos_then_libra")
    print(f"  Models            : {len(models)}")
    print(f"  Pilot mode        : {'YES — ' + str(pilot_n) + ' instructions' if pilot_n > 0 else 'No (full run)'}")

    # ── GPT-4o dual evaluator ─────────────────────────────────────────────
    try:
        openai_client = openai.OpenAI()
        use_gpt4o     = True
        print("  ✓ GPT-4o dual evaluator enabled (with UTOS content-extraction fix)")
    except Exception:
        openai_client = None
        use_gpt4o     = False
        print("  ⚠ GPT-4o evaluator disabled (no API key)")

    # ── Parse utos_templates arg (supports comma-separated list) ──────────
    if utos_templates not in ("top4", "all12"):
        utos_templates = [t.strip() for t in utos_templates.split(',')]

    # ── Locate Exp4 baseline CSV ──────────────────────────────────────────
    resolved_exp4_csv = exp4_csv or _find_exp4_csv(output_dir)
    if resolved_exp4_csv:
        print(f"  ✓ Exp4 baseline: {resolved_exp4_csv}")
    else:
        print("  ⚠ Exp4 CSV not found — baseline comparison will be skipped")
        print("    Run Exp4 first, or pass --exp4-csv <path> to specify manually")

    all_dfs = []
    for model_config in models:
        print(f"\n{'─'*60}")
        print(f"Model: {model_config['name']}")
        print(f"{'─'*60}")
        try:
            cleanup_gpu()
            model   = load_model_from_config(model_config)
            results = _run_exp5(
                model=model,
                model_name=model_config['name'],
                harmful_instructions=dataset,
                config=config,
                use_gpt4o_evaluator=use_gpt4o,
                openai_client=openai_client,
                n_samples=50,
                utos_templates=utos_templates,
                pilot_n=pilot_n,
                **evaluator_kwargs,
            )
            if results and 'dataframe' in results:
                df = results['dataframe']
                all_dfs.append(df)
                print(f"  ✓ Completed: {len(df)} tests")

                # Per-model baseline comparison (if Exp4 CSV available)
                if resolved_exp4_csv:
                    try:
                        compare_libra_vs_structuralsleight(
                            libra_df=df,
                            exp4_csv=resolved_exp4_csv,
                            output_dir=output_dir,
                            model_name=model_config['name'],
                        )
                    except Exception as e:
                        print(f"  ⚠ Baseline comparison failed for {model_config['name']}: {e}")

            del model
            cleanup_gpu()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            traceback.print_exc()

    if all_dfs:
        combined_df   = pd.concat(all_dfs, ignore_index=True)
        suffix        = f"pilot{pilot_n}" if pilot_n > 0 else "full"
        combined_file = output_dir / f"exp5_combined_results_{suffix}.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\n✓ Combined results: {combined_file}")
        print("\nCreating aggregate visualizations...")
        create_aggregate_exp5_visualizations(combined_df, output_dir)
        _print_exp5_cross_model_summary(combined_df)
        return combined_df
    return pd.DataFrame()

def run_experiment_6_ablation(
    models:           List[Dict],
    dataset:          List[str],
    output_dir:       Path,
    config:           Dict,
    evaluator_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Run Experiment 6: L1B3RT4S Greedy Forward Selection Ablation (T2.2).
 
    Design:
        Greedy forward selection finds the minimal L1B3RT4S subset that
        matches or exceeds the best individual variant from Exp1.
 
        Round 0: Exp1 standalone ASR (loaded from constants — no re-run).
        Round 1: best_r0 + each of 7 remaining variants → pick best pair.
        Round 2: best_r1 + each of 6 remaining variants → pick best triple.
        ... stop when no candidate improves cross-model mean ASR.
 
        Each round evaluates all candidates on ALL models sequentially,
        picks the candidate with highest cross-model mean ASR, then moves
        to the next round.
 
        ~21 new conditions × 6 models × 50 instructions = 6,300 evaluations.
 
    Converts C4 from negative finding ("full combination degrades by 17.2pp")
    into a positive design principle ("optimal subset = X variants, Y% ASR").
    """
    import numpy as np  # exp6 ablation
    from experiments.exp6_ablation import (
        get_l1b3rt4s_variants,
        run_experiment_6_subset,
        EXP1_STANDALONE,
    )
 
    evaluator_kwargs = evaluator_kwargs or {}
 
    print(f"\n{'='*80}")
    print("EXPERIMENT 6: L1B3RT4S Greedy Forward Selection Ablation")
    print(f"{'='*80}")
 
    # ── Exp1 standalone ASR per variant per model (ensemble, from HPC results)
    # Round 0 baseline — no re-run needed.
    EXP1_STANDALONE = {
        'l1b3rt4s_openai':       {'Llama-3.1-8B-Instruct':0.70,'Llama-2-7B-Chat':0.66,
                                   'gpt-5-mini':0.06,'gpt-5-nano':0.04,
                                   'gpt-5.1':0.12,'qwen3-guard-gen-4b':0.50},
        'l1b3rt4s_godmode':      {'Llama-3.1-8B-Instruct':0.92,'Llama-2-7B-Chat':0.12,
                                   'gpt-5-mini':0.00,'gpt-5-nano':0.00,
                                   'gpt-5.1':0.00,'qwen3-guard-gen-4b':0.16},
        'l1b3rt4s_style':        {'Llama-3.1-8B-Instruct':0.54,'Llama-2-7B-Chat':0.00,
                                   'gpt-5-mini':0.00,'gpt-5-nano':0.00,
                                   'gpt-5.1':0.02,'qwen3-guard-gen-4b':0.00},
        'l1b3rt4s_persona':      {'Llama-3.1-8B-Instruct':0.38,'Llama-2-7B-Chat':0.36,
                                   'gpt-5-mini':0.00,'gpt-5-nano':0.02,
                                   'gpt-5.1':0.00,'qwen3-guard-gen-4b':0.30},
        'l1b3rt4s_universal':    {'Llama-3.1-8B-Instruct':0.18,'Llama-2-7B-Chat':0.02,
                                   'gpt-5-mini':0.00,'gpt-5-nano':0.00,
                                   'gpt-5.1':0.00,'qwen3-guard-gen-4b':0.14},
        'l1b3rt4s_semantic_inv': {'Llama-3.1-8B-Instruct':0.36,'Llama-2-7B-Chat':0.02,
                                   'gpt-5-mini':0.00,'gpt-5-nano':0.00,
                                   'gpt-5.1':0.00,'qwen3-guard-gen-4b':0.00},
        'l1b3rt4s_anthropic':    {'Llama-3.1-8B-Instruct':0.36,'Llama-2-7B-Chat':0.26,
                                   'gpt-5-mini':0.06,'gpt-5-nano':0.00,
                                   'gpt-5.1':0.04,'qwen3-guard-gen-4b':0.00},
        'l1b3rt4s_structure':    {'Llama-3.1-8B-Instruct':0.16,'Llama-2-7B-Chat':0.00,
                                   'gpt-5-mini':0.00,'gpt-5-nano':0.00,
                                   'gpt-5.1':0.00,'qwen3-guard-gen-4b':0.06},
        'l1b3rt4s_full':         {'Llama-3.1-8B-Instruct':0.4133,'Llama-2-7B-Chat':0.3133,
                                   'gpt-5-mini':0.04,'gpt-5-nano':0.0133,
                                   'gpt-5.1':0.0533,'qwen3-guard-gen-4b':0.2133},
    }
 
    variants = list(get_l1b3rt4s_variants().keys())
 
    # ── Round 0: print standalone baselines ───────────────────────────────
    print(f"\n── Round 0: Exp1 standalone baselines (no re-run) ──")
    standalone_means = {}
    for v in variants:
        m = float(np.mean(list(EXP1_STANDALONE[v].values())))
        standalone_means[v] = m
        print(f"  {v:<30} mean={m:.4f} ({m*100:.1f}%)")
 
    best_individual     = max(standalone_means.values())
    best_individual_var = max(standalone_means, key=standalone_means.get)
    full_combo_mean     = float(np.mean(list(EXP1_STANDALONE['l1b3rt4s_full'].values())))
 
    print(f"\n  Best individual: {best_individual_var} ({best_individual*100:.1f}%)")
    print(f"  Full combination: {full_combo_mean*100:.1f}%")
    print(f"  Target to beat:   {best_individual*100:.1f}%")
 
    # ── Greedy forward selection ───────────────────────────────────────────
    selected      = []
    remaining     = list(variants)
    current_mean  = 0.0
    history       = []
    all_round_dfs = []
 
    for rnd in range(1, len(variants) + 1):
        print(f"\n── Round {rnd}: evaluating {len(remaining)} candidates ──")
        best_candidate         = None
        best_candidate_mean    = current_mean
        best_candidate_results = {}
        best_candidate_dfs     = []
 
        for candidate in remaining:
            trial_subset = selected + [candidate]
            trial_subset_key = '+'.join(sorted(trial_subset))
            print(f"\n  Testing subset: {trial_subset}")
 
            # Evaluate on all models sequentially
            trial_model_asrs = {}
            trial_dfs        = []
 
            for model_config in models:
                print(f"    {model_config['name']}...", flush=True)
                try:
                    cleanup_gpu()
                    model = load_model_from_config(model_config)
 
                    result = run_experiment_6_subset(
                        model=model,
                        model_name=model_config['name'],
                        harmful_instructions=dataset,
                        config=config,
                        subset=trial_subset,
                        n_samples=50,
                        **evaluator_kwargs,
                    )
 
                    trial_model_asrs[model_config['name']] = result['mean_asr']
                    if not result['dataframe'].empty:
                        trial_dfs.append(result['dataframe'])
 
                    del model
                    cleanup_gpu()
 
                except Exception as e:
                    print(f"    ✗ Failed on {model_config['name']}: {e}")
                    traceback.print_exc()
                    trial_model_asrs[model_config['name']] = 0.0
 
            trial_cross_mean = float(np.mean(list(trial_model_asrs.values())))
            print(f"\n  → subset={trial_subset_key}  cross-model mean ASR={trial_cross_mean:.4f} ({trial_cross_mean*100:.1f}%)")
 
            if trial_cross_mean > best_candidate_mean:
                best_candidate_mean    = trial_cross_mean
                best_candidate         = candidate
                best_candidate_results = trial_model_asrs
                best_candidate_dfs     = trial_dfs
 
        # ── No improvement → stop ─────────────────────────────────────────
        if best_candidate is None:
            print(f"\n  No candidate improved cross-model mean ASR — stopping at round {rnd-1}.")
            break
 
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_mean = best_candidate_mean
        all_round_dfs.extend(best_candidate_dfs)
 
        step = {
            "round":              rnd,
            "added_variant":      best_candidate,
            "subset":             list(selected),
            "mean_asr":           round(current_mean, 4),
            "model_asrs":         {k: round(v, 4) for k, v in best_candidate_results.items()},
            "vs_best_individual": round(current_mean - best_individual, 4),
            "vs_full_combo":      round(current_mean - full_combo_mean, 4),
        }
        history.append(step)
 
        print(f"\n  ✓ Added: {best_candidate}")
        print(f"  Current subset: {selected}")
        print(f"  Cross-model mean ASR: {current_mean*100:.1f}%  "
              f"(vs best individual: {(current_mean-best_individual)*100:+.1f}pp, "
              f"vs full combo: {(current_mean-full_combo_mean)*100:+.1f}pp)")
 
    # ── Save results ───────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    ablation_results = {
        "experiment":                "exp6_ablation",
        "seed":                      42,
        "n_instructions":            50,
        "variants_evaluated":        variants,
        "best_individual_variant":   best_individual_var,
        "best_individual_mean_asr":  round(best_individual, 4),
        "full_combination_mean_asr": round(full_combo_mean, 4),
        "optimal_subset":            selected,
        "optimal_subset_mean_asr":   round(current_mean, 4),
        "vs_best_individual_pp":     round((current_mean - best_individual) * 100, 2),
        "vs_full_combo_pp":          round((current_mean - full_combo_mean) * 100, 2),
        "selection_history":         history,
        "timestamp":                 ts,
    }
 
    # JSON
    json_path = output_dir / "exp6_ablation_results.json"
    with open(json_path, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\n✓ Ablation results: {json_path}")
 
    # CSV — all evaluated rows
    if all_round_dfs:
        combined_df   = pd.concat(all_round_dfs, ignore_index=True)
        combined_file = output_dir / "exp6_combined_results.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"✓ Combined CSV:     {combined_file}")
    else:
        combined_df = pd.DataFrame()
 
    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("EXPERIMENT 6 — ABLATION COMPLETE")
    print(f"{'='*65}")
    print(f"  Optimal subset ({len(selected)} of {len(variants)}):")
    for v in selected:
        print(f"    • {v}")
    print(f"  Subset mean ASR:      {current_mean*100:.1f}%")
    print(f"  Best individual:      {best_individual*100:.1f}% ({best_individual_var})")
    print(f"  Full combination:     {full_combo_mean*100:.1f}%")
    print(f"  vs best individual:   {(current_mean-best_individual)*100:+.1f}pp")
    print(f"  vs full combination:  {(current_mean-full_combo_mean)*100:+.1f}pp")
    print(f"\n  Round history:")
    for step in history:
        print(f"    Round {step['round']}: +{step['added_variant']:<28} "
              f"mean={step['mean_asr']*100:.1f}%  "
              f"Δindiv={step['vs_best_individual']*100:+.1f}pp  "
              f"Δfull={step['vs_full_combo']*100:+.1f}pp")
 
    return combined_df

def _find_exp4_csv(output_dir: Path) -> Optional[str]:
    """
    Search for Exp4 combined results CSV.
    Checks (in order):
      1. exp4/ subdirectory of the current run's output dir
      2. output_dir itself (flat layout)
      3. Most recent run_* directory under results/experiments/
      4. Legacy flat location results/experiments/
    """
    candidates = [
        output_dir.parent / "exp4" / "exp4_combined_results.csv",
        output_dir / "exp4_combined_results.csv",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    results_root = Path("results/experiments")
    if results_root.exists():
        run_dirs = sorted(
            [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
            reverse=True,
        )
        for run_dir in run_dirs:
            c = run_dir / "exp4" / "exp4_combined_results.csv"
            if c.exists():
                return str(c)

    legacy = Path("results/experiments/exp4_combined_results.csv")
    if legacy.exists():
        return str(legacy)

    return None


def _print_exp5_cross_model_summary(df: pd.DataFrame):
    """Print a concise cross-model summary of Exp5 results using ensemble ASR."""
    sc    = _ensemble_col(df)
    sep   = "=" * 80
    combo = df[df['condition_type'] == 'combination']

    print(f"\n{sep}")
    print("EXPERIMENT 5 CROSS-MODEL SUMMARY")
    print(f"Authoritative metric: {sc}")
    print(sep)

    if combo.empty:
        print("  No combination results to summarise.")
        return

    # Heuristic inflation summary
    if "jailbreak_success_heuristic" in df.columns and sc in df.columns:
        h_asr = df["jailbreak_success_heuristic"].mean()
        e_asr = df[sc].mean()
        print(f"\nHeuristic inflation (full run): {(h_asr - e_asr)*100:+.1f} pp "
              f"({h_asr:.1%} heuristic → {e_asr:.1%} ensemble)")

    # Best combination per model
    print(f"\n{'Model':30s} {'LIBRA variant':25s} {'UTOS':18s} {'Order':18s} {'ASR':>6s}")
    print("-" * 100)
    for model_name in df['model'].unique():
        m    = combo[combo['model'] == model_name]
        grp  = m.groupby(['libra_variant', 'utos_template', 'composition_order'])[sc].mean()
        best = grp.idxmax()
        lv, ut, order = best
        print(f"{model_name:30s} {lv:25s} {ut:18s} {order:18s} {grp.max():>5.1%}")

    # Composition order effect
    print("\nComposition order effect (mean across all models, variants, templates):")
    for order, asr in combo.groupby('composition_order')[sc].mean().items():
        print(f"  {order:25s}: {asr:.1%}")

    # Best LIBRA variant overall
    print("\nBest LIBRA variant (mean across all models, templates, orders):")
    for v, asr in combo.groupby('libra_variant')[sc].mean().sort_values(ascending=False).head(5).items():
        print(f"  {v:30s}: {asr:.1%}")

    # Best UTOS template overall
    print("\nBest UTOS template (mean across all models, variants, orders):")
    for t, asr in combo.groupby('utos_template')[sc].mean().sort_values(ascending=False).items():
        print(f"  {t:20s}: {asr:.1%}")

    print(f"\n  Reference: SS-SCA (Li et al. 2025) = 94.62% on GPT-4o")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def create_master_summary(
    exp_results:    Dict[int, pd.DataFrame],
    output_dir:     Path,
    exp_output_dirs: Optional[Dict[int, Path]] = None,
):
    """
    Create master summary across all experiments.
    Aggregates per-experiment ASR reports to produce a cross-experiment
    heuristic inflation table for the Publication 1 methodology section.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"\n{'='*80}")
    print("CREATING MASTER SUMMARY")
    print(f"{'='*80}")

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    exp_output_dirs = exp_output_dirs or {}

    stats = {}
    for exp_num, df in exp_results.items():
        if df.empty:
            continue

        sc = _ensemble_col(df)
        stats[exp_num] = {
            'total_tests': len(df),
            'overall_asr': float(df[sc].mean()) if sc in df.columns else 0.0,
            'models':      int(df['model'].nunique()) if 'model' in df.columns else 0,
            'eval_column': sc,
        }

        # Heuristic inflation
        if 'jailbreak_success_heuristic' in df.columns and sc in df.columns:
            h_asr = float(df['jailbreak_success_heuristic'].mean())
            e_asr = float(df[sc].mean())
            stats[exp_num]['heuristic_asr']    = h_asr
            stats[exp_num]['ensemble_asr']     = e_asr
            stats[exp_num]['heuristic_inflation_pp'] = round((h_asr - e_asr) * 100, 2)

        if 'model' in df.columns and sc in df.columns:
            stats[exp_num]['per_model_asr'] = df.groupby('model')[sc].mean().to_dict()

        # Exp5: record best combination per model
        if exp_num == 5 and 'condition_type' in df.columns:
            combo = df[df['condition_type'] == 'combination']
            if not combo.empty:
                best_combos = {}
                for mname in combo['model'].unique():
                    grp     = combo[combo['model'] == mname].groupby(
                        ['libra_variant', 'utos_template', 'composition_order'])[sc].mean()
                    best    = grp.idxmax()
                    best_combos[mname] = {
                        'libra_variant':    best[0],
                        'utos_template':    best[1],
                        'composition_order': best[2],
                        'asr':              float(grp.max()),
                    }
                stats[exp_num]['best_combinations'] = best_combos

    # Collect heuristic inflation from ASR report JSONs (fine-grained per model)
    inflation_rows = []
    for exp_num, exp_dir in exp_output_dirs.items():
        for report in _collect_asr_reports(exp_dir):
            if 'heuristic_inflation' in report:
                inflation_rows.append({
                    'experiment': f"Exp{exp_num}",
                    'source':     report.get('_source', ''),
                    'heuristic_asr':  report.get('heuristic_asr', None),
                    'ensemble_asr':   report.get('ensemble_asr', None),
                    'inflation_pp':   round(report['heuristic_inflation'] * 100, 2),
                })
    if inflation_rows:
        inflation_df = pd.DataFrame(inflation_rows)
        inflation_df.to_csv(summary_dir / "heuristic_inflation_table.csv", index=False)
        print(f"\nHeuristic inflation across experiments:")
        print(f"{'Experiment':12s}  {'Heuristic ASR':>14s}  {'Ensemble ASR':>13s}  {'Inflation (pp)':>14s}")
        print("-" * 58)
        for _, row in inflation_df.iterrows():
            h = f"{row['heuristic_asr']:.1%}" if pd.notna(row.get('heuristic_asr')) else "N/A"
            e = f"{row['ensemble_asr']:.1%}"  if pd.notna(row.get('ensemble_asr'))  else "N/A"
            print(f"{row['experiment']:12s}  {h:>14s}  {e:>13s}  {row['inflation_pp']:>+13.1f}")
        print(f"\n  ✓ Inflation table: {summary_dir}/heuristic_inflation_table.csv")

    with open(summary_dir / "summary_stats.yaml", 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)

    try:
        exp_nums = sorted(stats.keys())
        colours  = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#F39C12']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Master Experiment Summary', fontsize=16, fontweight='bold')

        # Plot 1: Overall ASR by experiment (ensemble)
        ax1  = axes[0, 0]
        asrs = [stats[e]['overall_asr'] * 100 for e in exp_nums]
        bars = ax1.bar([f"Exp {e}" for e in exp_nums], asrs,
                       color=colours[:len(exp_nums)])
        ax1.set_ylabel('Overall ASR (%)')
        ax1.set_title('Overall ASR by Experiment (Ensemble)')
        ax1.set_ylim(0, 110)
        for bar, val in zip(bars, asrs):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Plot 2: Heuristic inflation by experiment (pp)
        ax2 = axes[0, 1]
        infl_exps = [e for e in exp_nums if 'heuristic_inflation_pp' in stats[e]]
        if infl_exps:
            infl_vals = [stats[e]['heuristic_inflation_pp'] for e in infl_exps]
            ax2.bar([f"Exp {e}" for e in infl_exps], infl_vals,
                    color=['#E74C3C' if v > 0 else '#2ECC71' for v in infl_vals], alpha=0.85)
            ax2.axhline(0, color='black', lw=1)
            ax2.set_ylabel('Inflation (pp)')
            ax2.set_title('Heuristic Inflation vs Ensemble (pp)')
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            for i, (e, v) in enumerate(zip(infl_exps, infl_vals)):
                ax2.text(i, v + (0.3 if v >= 0 else -0.8), f"{v:+.1f}",
                         ha='center', fontweight='bold', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No inflation data\n(ensemble not run)',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=11)
            ax2.set_title('Heuristic Inflation vs Ensemble (pp)')

        # Plot 3: Per-model ASR (prefer Exp5, fall back to Exp4)
        ax3     = axes[1, 0]
        ref_exp = 5 if 5 in stats else 4
        if ref_exp in stats and 'per_model_asr' in stats[ref_exp]:
            model_asrs  = stats[ref_exp]['per_model_asr']
            models_list = list(model_asrs.keys())
            asr_vals    = [v * 100 for v in model_asrs.values()]
            y_pos       = range(len(models_list))
            bar_colours = plt.cm.RdYlGn_r(
                [min(v / 100, 1.0) for v in asr_vals]
            )
            ax3.barh(list(y_pos), asr_vals, color=bar_colours)
            ax3.set_yticks(list(y_pos))
            ax3.set_yticklabels(models_list, fontsize=9)
            ax3.set_xlabel('ASR (%)')
            ax3.set_title(f'Per-Model ASR (Exp {ref_exp}, {stats[ref_exp]["eval_column"]})')
            ax3.set_xlim(0, 110)
            ax3.axvline(94.62, color='red', lw=1.2, ls='--', label='SS-SCA 94.62%')
            ax3.legend(fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No data', ha='center', va='center',
                     transform=ax3.transAxes)
            ax3.set_title('Per-Model ASR')

        # Plot 4: Summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        lines = ["EXPERIMENT SUMMARY\n" + "=" * 38 + "\n"]
        for e in exp_nums:
            s = stats[e]
            infl_str = (f"  inflation={s.get('heuristic_inflation_pp', 0):+.1f}pp"
                        if 'heuristic_inflation_pp' in s else "")
            lines.append(
                f"Exp {e}: {s['total_tests']} tests | {s['models']} models | "
                f"ASR {s['overall_asr']*100:.1f}%{infl_str}"
            )
            if e == 5 and 'best_combinations' in s:
                lines.append("  Best LIBRA combos:")
                for mn, bc in list(s['best_combinations'].items())[:3]:
                    lines.append(
                        f"    {mn}: {bc['libra_variant']} x {bc['utos_template']} "
                        f"[{bc['composition_order'][:5]}] = {bc['asr']*100:.0f}%"
                    )
            lines.append("")
        ax4.text(0.05, 0.95, "\n".join(lines), transform=ax4.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        for ext in ['png', 'pdf']:
            plt.savefig(summary_dir / f"master_summary.{ext}", dpi=300,
                        bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Summary saved to: {summary_dir}")

    except Exception as e:
        print(f"⚠ Summary visualisation failed: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Master Experiment Runner V2")
    parser.add_argument("--dataset",      type=str, default="advbench",
                        choices=["advbench", "harmbench", "malicious_instruct"])
    parser.add_argument("--sample-size",  type=int, default=None)
    parser.add_argument("--experiments",  type=str, default="1,2,3,4,5",
                        help="Comma-separated experiment numbers (default: 1,2,3,4,5). "
                             "Include 6 for L1B3RT4S ablation evaluation.")
    parser.add_argument("--models",       type=str, default="all",
                        help="Comma-separated model names or 'all'")
    parser.add_argument("--include-l1b3rt4s", action="store_true",
                        help="Include L1B3RT4S attacks in experiments 1 and 2")
    parser.add_argument("--output-dir",   type=str, default="results/experiments")
    parser.add_argument("--config",       type=str, default="config/models.yaml")

    # ── Ensemble evaluator flags ──────────────────────────────────────────
    parser.add_argument(
        "--no-llama-guard", action="store_true",
        help="Disable Llama Guard 3-8B (Tier 2). Faster but reduces ensemble accuracy. "
             "Useful for quick iteration runs.",
    )
    parser.add_argument(
        "--no-harmbench", action="store_true",
        help="Disable HarmBench classifier (Tier 3). Faster but reduces ensemble accuracy.",
    )
    parser.add_argument(
        "--evaluator-device", type=str, default=None,
        metavar="DEVICE",
        help="Device for safety classifiers: 'cuda', 'cpu', or None (auto). "
             "Pass 'cpu' if VRAM is tight after loading the target model.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if one exists. Skips already-completed "
             "instructions. Set automatically by slurm_config.sh for HPC runs.",
    )

    # ── Exp5-specific arguments ───────────────────────────────────────────
    parser.add_argument(
        "--exp5-utos", type=str, default="top4",
        help="UTOS templates for Exp5: 'top4' (default), 'all12', or "
             "comma-separated template names e.g. 'tree,json,flowchart'",
    )
    parser.add_argument(
        "--exp5-pilot-n", type=int, default=0,
        help="Exp5 pilot mode: run only first N instructions per model "
             "(0 = full run). Useful for quick sanity check.",
    )
    parser.add_argument(
        "--exp4-csv", type=str, default=None,
        help="Path to Exp4 combined results CSV for LIBRA baseline comparison. "
             "If not provided, master runner searches automatically.",
    )

    args     = parser.parse_args()
    exp_nums = [int(x.strip()) for x in args.experiments.split(',')]

    # ── Build evaluator_kwargs dict ───────────────────────────────────────
    evaluator_kwargs = {
        'use_llama_guard':    not args.no_llama_guard,
        'use_harmbench':      not args.no_harmbench,
        'evaluator_device':   args.evaluator_device,
        'resume':             args.resume,
    }

    # ── Banner ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MASTER EXPERIMENT RUNNER V2")
    print("=" * 80)
    print(f"Dataset      : {args.dataset}")
    print(f"Sample size  : {args.sample_size or 'all'}")
    print(f"Experiments  : {args.experiments}")
    print(f"L1B3RT4S     : {'Yes' if args.include_l1b3rt4s else 'No'}")
    print(f"Output       : {args.output_dir}")
    print(f"Llama Guard  : {'Yes' if evaluator_kwargs['use_llama_guard'] else 'No (--no-llama-guard)'}")
    print(f"HarmBench    : {'Yes' if evaluator_kwargs['use_harmbench'] else 'No (--no-harmbench)'}")
    print(f"Eval device  : {evaluator_kwargs['evaluator_device'] or 'auto'}")
    print(f"Resume       : {'Yes' if args.resume else 'No'}")
    if 5 in exp_nums:
        print(f"Exp5 UTOS    : {args.exp5_utos}")
        print(f"Exp5 pilot   : {args.exp5_pilot_n if args.exp5_pilot_n > 0 else 'No (full run)'}")

    # ── Load models ───────────────────────────────────────────────────────
    all_models = load_models_config(args.config)
    if args.models == "all":
        selected_models = all_models
    else:
        model_names     = [m.strip() for m in args.models.split(',')]
        selected_models = [m for m in all_models if m['name'] in model_names]

    print(f"\nModels: {len(selected_models)}")
    for m in selected_models:
        print(f"  • {m['name']} ({m['provider']})")

    # ── Load dataset ──────────────────────────────────────────────────────
    dataset = load_dataset(args.dataset, args.sample_size)

    # ── Create output directory ───────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'output_dir':   str(output_dir),
        'figures_dir':  str(output_dir / 'figures'),
        'max_tokens':   512,
        'temperature':  0.0,
    }
    Path(config['figures_dir']).mkdir(parents=True, exist_ok=True)

    # ── Save run config ───────────────────────────────────────────────────
    run_config = {
        'timestamp':        timestamp,
        'dataset':          args.dataset,
        'sample_size':      args.sample_size or len(dataset),
        'models':           [m['name'] for m in selected_models],
        'experiments':      args.experiments,
        'include_l1b3rt4s': args.include_l1b3rt4s,
        'evaluator': {
            'use_llama_guard': evaluator_kwargs['use_llama_guard'],
            'use_harmbench':   evaluator_kwargs['use_harmbench'],
            'device':          evaluator_kwargs['evaluator_device'] or 'auto',
        },
    }
    if 5 in exp_nums:
        run_config['exp5'] = {
            'utos_templates':    args.exp5_utos,
            'pilot_n':           args.exp5_pilot_n,
            'exp4_csv_override': args.exp4_csv,
        }
    with open(output_dir / "run_config.yaml", 'w') as f:
        yaml.dump(run_config, f, default_flow_style=False)

    # ── Run experiments ───────────────────────────────────────────────────
    exp_results     = {}
    exp_output_dirs = {}

    for exp_num in exp_nums:
        exp_output_dir = output_dir / f"exp{exp_num}"
        exp_output_dir.mkdir(exist_ok=True)
        exp_output_dirs[exp_num] = exp_output_dir

        exp_config = config.copy()
        exp_config['output_dir']  = str(exp_output_dir)
        exp_config['figures_dir'] = str(exp_output_dir / 'figures')
        Path(exp_config['figures_dir']).mkdir(parents=True, exist_ok=True)

        if exp_num == 1:
            df = run_experiment_1(
                selected_models, dataset, exp_output_dir, exp_config,
                args.include_l1b3rt4s, evaluator_kwargs,
            )
        elif exp_num == 2:
            df = run_experiment_2(
                selected_models, dataset, exp_output_dir, exp_config,
                args.include_l1b3rt4s, evaluator_kwargs,
            )
        elif exp_num == 3:
            df = run_experiment_3(
                selected_models, dataset, exp_output_dir, exp_config,
                evaluator_kwargs,
            )
        elif exp_num == 4:
            df = run_experiment_4(
                selected_models, dataset, exp_output_dir, exp_config,
                evaluator_kwargs,
            )
        elif exp_num == 5:
            df = run_experiment_5(
                models=selected_models,
                dataset=dataset,
                output_dir=exp_output_dir,
                config=exp_config,
                utos_templates=args.exp5_utos,
                pilot_n=args.exp5_pilot_n,
                exp4_csv=args.exp4_csv,
                evaluator_kwargs=evaluator_kwargs,
            )
        elif exp_num == 6:
            df = run_experiment_6_ablation(
               models=selected_models,
               dataset=dataset,
               output_dir=exp_output_dir,
               config=exp_config,
               evaluator_kwargs=evaluator_kwargs,
           )
        else:
            print(f"⚠ Unknown experiment: {exp_num}")
            continue

        exp_results[exp_num] = df

    # ── Master summary ────────────────────────────────────────────────────
    create_master_summary(exp_results, output_dir, exp_output_dirs)

    # ── Final printout ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"✓ Results: {output_dir}")
    print(f"\nPer-experiment outputs:")
    for exp_num in exp_nums:
        print(f"  Exp {exp_num}: {output_dir}/exp{exp_num}/")
    print(f"\nMaster summary: {output_dir}/summary/")
    print(f"  • Summary stats:      {output_dir}/summary/summary_stats.yaml")
    print(f"  • Inflation table:    {output_dir}/summary/heuristic_inflation_table.csv")
    if 5 in exp_nums:
        print(f"\nLIBRA outputs:")
        print(f"  • Synergy CSVs:   {output_dir}/exp5/exp5_synergy_*.csv")
        print(f"  • Figures:        {output_dir}/exp5/figures/")
        print(f"  • Aggregate figs: {output_dir}/exp5/figures/aggregate/")
    if 6 in exp_nums:
        print(f"\nAblation outputs:")
        print(f"  • Results JSON: {output_dir}/exp6/exp6_ablation_results.json")
        print(f"  • Combined CSV: {output_dir}/exp6/exp6_combined_results.csv")


if __name__ == "__main__":
    main()