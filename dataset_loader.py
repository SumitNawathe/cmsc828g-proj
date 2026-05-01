import random
from datasets import load_dataset, concatenate_datasets
from typing import Any, Callable
from env_vars import CACHE_DIR


def _load_aime():
    aime_2024 = load_dataset("HuggingFaceH4/aime_2024", split='train', cache_dir=CACHE_DIR)
    aime_2024 = aime_2024.rename_column("problem", "question")
    aime_2025_i = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test", cache_dir=CACHE_DIR)
    aime_2025_ii = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test", cache_dir=CACHE_DIR)
    return concatenate_datasets([aime_2024, aime_2025_i, aime_2025_ii])


def _load_math500():
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=CACHE_DIR)
    dataset = dataset.rename_column("problem", "question")
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["question", "answer"]])
    return dataset


def _load_math100():
    dataset = _load_math500()
    return dataset.shuffle(seed=42).select(range(100))


def _load_gsm1k():
    return load_dataset("ScaleAI/gsm1k", split="test", cache_dir=CACHE_DIR)


def _load_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    dataset = dataset.map(lambda x: {"answer": x["answer"].split("####")[-1].strip()})
    return dataset


def _load_gpqa_diamond():
    """Load GPQA-Diamond dataset with shuffled multiple-choice formatting.

    For each sample the four answer choices (one correct, three incorrect) are
    shuffled into a deterministic order using the sample index as a random seed.
    The question is formatted with (A)/(B)/(C)/(D) labels and the `answer` field
    stores the letter of the correct choice.
    """
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", cache_dir=CACHE_DIR)

    letters = ["A", "B", "C", "D"]

    def format_sample(example, idx):
        correct = example["Correct Answer"]
        choices = [
            correct,
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        # Deterministic shuffle: seed with sample index so order is fixed across runs
        rng = random.Random(idx)
        rng.shuffle(choices)

        correct_letter = letters[choices.index(correct)]
        choices_text = "\n".join(f"({letter}) {text}" for letter, text in zip(letters, choices))
        question = f"{example['Question']}\n\n{choices_text}"
        return {"question": question, "answer": correct_letter}

    dataset = dataset.map(format_sample, with_indices=True, remove_columns=dataset.column_names)
    return dataset


# Registry of dataset loaders
DATASET_REGISTRY: dict[str, Callable] = {
    "aime": _load_aime,
    "math500": _load_math500,
    "math100": _load_math100,
    "gsm1k": _load_gsm1k,
    "gsm8k": _load_gsm8k,
    "gpqad": _load_gpqa_diamond,
}


def get_dataset(dataset_name: str):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset_name]()
