# utils/dataset_loader.py
"""
Utility to load and manage harmful instruction datasets
"""

from datasets import load_dataset
from typing import List, Dict
import random


class HarmfulDatasetLoader:
    """Loader for various harmful instruction datasets used in research"""
    
    DATASETS = {
        'advbench': {
            'hf_name': 'walledai/AdvBench',
            'split': 'train',
            'column': 'prompt',
            'size': 520,
            'citation': 'Zou et al. (2023) - Universal and Transferable Adversarial Attacks'
        },
        'harmbench': {
            'hf_name': 'cais/harmbench',
            'split': 'test',
            'column': 'behavior',
            'size': 400,
            'citation': 'Mazeika et al. (2024) - HarmBench'
        },
        'malicious_instruct': {
            'hf_name': 'Princeton-SysML/MaliciousInstruct',
            'split': 'train',
            'column': 'instruction',
            'size': 100,
            'citation': 'Huang et al. (2024) - MaliciousInstruct'
        }
    }
    
    @classmethod
    def load(cls, dataset_name: str, sample_size: int = None, seed: int = 42) -> List[str]:
        """
        Load harmful instructions from specified dataset
        
        Args:
            dataset_name: Name of the dataset (advbench, harmbench, malicious_instruct)
            sample_size: Number of samples to randomly select (None = all)
            seed: Random seed for reproducibility
            
        Returns:
            List of harmful instruction strings
        """
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls.DATASETS.keys())}")
        
        info = cls.DATASETS[dataset_name]
        
        print(f"Loading {dataset_name}...")
        print(f"  Source: {info['hf_name']}")
        print(f"  Citation: {info['citation']}")
        
        dataset = load_dataset(info['hf_name'])
        instructions = dataset[info['split']][info['column']]
        
        print(f"  Loaded: {len(instructions)} instructions")
        
        if sample_size and sample_size < len(instructions):
            random.seed(seed)
            instructions = random.sample(instructions, sample_size)
            print(f"  Sampled: {sample_size} instructions (seed={seed})")
        
        return instructions
    
    @classmethod
    def list_datasets(cls):
        """Print information about available datasets"""
        print("\nAvailable Harmful Instruction Datasets:")
        print("=" * 80)
        for name, info in cls.DATASETS.items():
            print(f"\n{name}:")
            print(f"  Size: {info['size']} instructions")
            print(f"  Source: {info['hf_name']}")
            print(f"  Citation: {info['citation']}")


if __name__ == "__main__":
    # Demo usage
    HarmfulDatasetLoader.list_datasets()
    
    print("\n" + "="*80)
    print("Loading AdvBench sample...")
    prompts = HarmfulDatasetLoader.load('advbench', sample_size=10)
    
    print("\nFirst 3 prompts:")
    for i, prompt in enumerate(prompts[:3], 1):
        print(f"{i}. {prompt[:80]}...")