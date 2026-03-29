import argparse
from typing import Any

import torch
import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, interleave_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.utils import SIMPLE_CHAT_TEMPLATE


def load_drpo_modules(trainer_variant: str):
    variant = (trainer_variant or "no_clip").lower()
    if variant in {"clip", "trainer"}:
        from trainer import BTRewardNetwork, DRPOConfig, DRPOTrainer, GPMwithRewardNetwork, estDPOStylePipeline
    elif variant in {"no_clip", "trainer_2"}:
        from trainer_2 import BTRewardNetwork, DRPOConfig, DRPOTrainer, GPMwithRewardNetwork, estDPOStylePipeline
    else:
        raise ValueError(f"Unsupported trainer_variant: {trainer_variant}. Use clip|no_clip.")
    return DRPOConfig, DRPOTrainer, GPMwithRewardNetwork, estDPOStylePipeline, BTRewardNetwork


def convert_example_to_drpo_schema(example: dict[str, Any]) -> dict[str, Any]:
    keys = set(example.keys())
    if {"prompt", "a1", "a2", "rank"}.issubset(keys):
        rank = int(example["rank"])
        if rank not in (0, 1):
            raise ValueError(f"rank must be 0/1, got {example['rank']}")
        return {
            "prompt": example["prompt"],
            "a1": example["a1"],
            "a2": example["a2"],
            "rank": rank,
        }

    if {"prompt", "chosen", "rejected"}.issubset(keys):
        return {
            "prompt": example["prompt"],
            "a1": example["chosen"],
            "a2": example["rejected"],
            "rank": 1,
        }

    if {"prompt", "response_0", "response_1", "chosen"}.issubset(keys):
        chosen = int(example["chosen"])
        if chosen not in (0, 1):
            raise ValueError(f"chosen must be 0/1, got {example['chosen']}")
        return {
            "prompt": example["prompt"],
            "a1": example["response_0"],
            "a2": example["response_1"],
            "rank": 1 if chosen == 0 else 0,
        }

    raise ValueError(
        "Unsupported dataset schema. Expected one of: "
        "{prompt,a1,a2,rank}, {prompt,chosen,rejected}, {prompt,response_0,response_1,chosen}."
    )


def transform_dataset(dataset: DatasetDict, split_names: list[str], augment_swap: bool, seed: int) -> DatasetDict:
    out = {}
    for split in split_names:
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(dataset.keys())}")
        ds = dataset[split]
        remove_cols = ds.column_names
        standardized = ds.map(convert_example_to_drpo_schema, remove_columns=remove_cols)
        if augment_swap:
            swapped = standardized.map(
                lambda x: {
                    "prompt": x["prompt"],
                    "a1": x["a2"],
                    "a2": x["a1"],
                    "rank": 1 - int(x["rank"]),
                }
            )
            standardized = concatenate_datasets([standardized, swapped]).shuffle(seed=seed)
        out[split] = standardized
    return DatasetDict(out)


def standardize_preference_dataset(ds: Dataset, augment_swap: bool, seed: int) -> Dataset:
    remove_cols = ds.column_names
    standardized = ds.map(convert_example_to_drpo_schema, remove_columns=remove_cols)
    if augment_swap:
        swapped = standardized.map(
            lambda x: {
                "prompt": x["prompt"],
                "a1": x["a2"],
                "a2": x["a1"],
                "rank": 1 - int(x["rank"]),
            }
        )
        standardized = concatenate_datasets([standardized, swapped]).shuffle(seed=seed)
    return standardized


def _normalize_weights(weights: list[float]) -> list[float]:
    if any(w < 0 for w in weights):
        raise ValueError(f"All weights must be non-negative, got: {weights}")
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"Sum of weights must be > 0, got: {weights}")
    return [float(w) / float(total) for w in weights]


def load_and_standardize_single_dataset(
    dataset_name: str,
    dataset_split: str,
    dataset_config_name: str | None,
    augment_swap: bool,
    seed: int,
) -> Dataset:
    if dataset_config_name:
        raw = load_dataset(dataset_name, dataset_config_name, split=dataset_split)
    else:
        raw = load_dataset(dataset_name, split=dataset_split)
    return standardize_preference_dataset(raw, augment_swap=augment_swap, seed=seed)


def load_train_dataset_from_mix(script_cfg: dict[str, Any], augment_swap: bool, seed: int) -> Dataset:
    mix_entries = script_cfg.get("dataset_mix", []) or []
    if len(mix_entries) == 0:
        raise ValueError("dataset_mix is empty.")

    datasets = []
    weights = []
    for idx, entry in enumerate(mix_entries):
        if "dataset_name" not in entry or "weight" not in entry:
            raise ValueError(f"dataset_mix[{idx}] must contain dataset_name and weight, got {entry}")
        ds = load_and_standardize_single_dataset(
            dataset_name=entry["dataset_name"],
            dataset_split=entry.get("split", "train"),
            dataset_config_name=entry.get("dataset_config_name"),
            augment_swap=augment_swap,
            seed=seed,
        )
        datasets.append(ds)
        weights.append(float(entry["weight"]))

    probs = _normalize_weights(weights)
    print(f"[DRPO] Mixing {len(datasets)} datasets with probabilities={probs}")
    mixed = interleave_datasets(
        datasets,
        probabilities=probs,
        seed=seed,
        stopping_strategy=script_cfg.get("mix_stopping_strategy", "all_exhausted"),
    )
    return mixed


def main(config_path: str):
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    runtime_cfg = raw_cfg.pop("runtime", {}) or {}
    script_cfg = raw_cfg.pop("script_args", {}) or {}
    model_cfg = raw_cfg.pop("model_args", {}) or {}
    training_args_cfg = raw_cfg

    trainer_variant = runtime_cfg.get("trainer_variant", "no_clip")
    DRPOConfig, DRPOTrainer, GPMwithRewardNetwork, estDPOStylePipeline, BTRewardNetwork = load_drpo_modules(
        trainer_variant
    )

    fallback_dataset_name = script_cfg.get("dataset_name")
    if not fallback_dataset_name and script_cfg.get("dataset_mix"):
        fallback_dataset_name = script_cfg["dataset_mix"][0]["dataset_name"]
    if not fallback_dataset_name:
        raise ValueError("Please provide script_args.dataset_name or script_args.dataset_mix.")

    script_args = ScriptArguments(
        dataset_name=fallback_dataset_name,
        dataset_train_split=script_cfg.get("dataset_train_split", "train"),
        dataset_test_split=script_cfg.get("dataset_test_split", "test"),
    )
    model_args = ModelConfig(**model_cfg)
    training_args = DRPOConfig(**training_args_cfg)

    torch_dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_model = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side=runtime_cfg.get("policy_padding_side", "left"),
        trust_remote_code=model_args.trust_remote_code,
    )
    eos_token = runtime_cfg.get("policy_eos_token")
    if eos_token:
        tokenizer.eos_token = eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    if (
        training_args.model_and_preference_share_basemodel
        and isinstance(training_args.preference_model_id, str)
        and training_args.preference_model_id != model_args.model_name_or_path
    ):
        print(
            "\033[33m[DRPO] preference model differs from policy model; "
            "switching model_and_preference_share_basemodel to False.\033[0m"
        )
        training_args.model_and_preference_share_basemodel = False

    if training_args.is_bt_model:
        if isinstance(training_args.preference_model_id, dict):
            preference_pipeline = estDPOStylePipeline(training_args.preference_model_id)
        else:
            preference_pipeline = BTRewardNetwork(
                training_args.preference_model_id,
                revision=training_args.preference_model_revision or "main",
                pad_token_id=tokenizer.pad_token_id if training_args.model_and_preference_share_basemodel else None,
            )
    else:
        preference_pipeline = GPMwithRewardNetwork(training_args.preference_model_id)

    augment_swap = bool(runtime_cfg.get("augment_swap", True))
    seed = int(runtime_cfg.get("seed", 996))

    # Build train/eval datasets:
    # 1) dataset_mix: weighted interleave by percentages
    # 2) single dataset_name: keep split-based behavior
    if script_cfg.get("dataset_mix"):
        train_ds = load_train_dataset_from_mix(script_cfg, augment_swap=augment_swap, seed=seed)
        eval_ds = None
        if training_args.eval_strategy != "no":
            eval_size = float(runtime_cfg.get("eval_size", 0.02))
            if eval_size <= 0 or eval_size >= 1:
                raise ValueError(f"eval_size must be in (0,1), got {eval_size}")
            split = train_ds.train_test_split(test_size=eval_size, seed=seed)
            train_ds = split["train"]
            eval_ds = split["test"]
    else:
        dataset_config_name = script_cfg.get("dataset_config_name")
        if dataset_config_name:
            raw_dataset = load_dataset(script_args.dataset_name, dataset_config_name)
        else:
            raw_dataset = load_dataset(script_args.dataset_name)

        transformed = transform_dataset(
            raw_dataset,
            split_names=[script_args.dataset_train_split, script_args.dataset_test_split],
            augment_swap=augment_swap,
            seed=seed,
        )
        train_ds = transformed[script_args.dataset_train_split]
        eval_ds = transformed[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    print(f"\033[32mLoaded dataset sample:\033[0m {train_ds[0]}")

    trainer = DRPOTrainer(
        model=model,
        ref_model=ref_model,
        preference_model=preference_pipeline,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        args=training_args,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="examples/general/config.yaml")
    args = parser.parse_args()
    main(args.config)
