import argparse
from collections import defaultdict
from typing import Any

import yaml
import torch
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


def _messages_to_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages)
    chunks = []
    for m in messages:
        if isinstance(m, dict):
            role = str(m.get("role", "unknown"))
            content = str(m.get("content", ""))
            chunks.append(f"{role}: {content}")
        else:
            chunks.append(str(m))
    return "\n".join(chunks).strip()


def _to_plain_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return _messages_to_text(x)
    if isinstance(x, dict):
        role = str(x.get("role", "unknown"))
        content = str(x.get("content", ""))
        return f"{role}: {content}".strip()
    return str(x)


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
    prompt_key = None
    for k in ["prompt", "instruction", "input", "question"]:
        if k in keys:
            prompt_key = k
            break

    def _get_prompt():
        if prompt_key is None:
            return None
        return example[prompt_key]
    if {"a1", "a2", "rank"}.issubset(keys) and prompt_key is not None:
        rank = int(example["rank"])
        if rank not in (0, 1):
            raise ValueError(f"rank must be 0/1, got {example['rank']}")
        return {
            "prompt": _get_prompt(),
            "a1": example["a1"],
            "a2": example["a2"],
            "rank": rank,
        }

    if {"chosen", "rejected"}.issubset(keys) and prompt_key is not None:
        # UltraFeedback-style conversational records: chosen/rejected are full dialogs
        # [user, assistant]. Convert to prompt + assistant completion pair.
        if isinstance(example["chosen"], list) and isinstance(example["rejected"], list):
            chosen = example["chosen"]
            rejected = example["rejected"]
            if len(chosen) >= 2 and len(rejected) >= 2:
                prompt_msgs = chosen[:-1]
                a1_msg = chosen[-1:]
                a2_msg = rejected[-1:]
                return {
                    "prompt": prompt_msgs,
                    "a1": a1_msg,
                    "a2": a2_msg,
                    "rank": 1,
                }
        return {
            "prompt": _get_prompt(),
            "a1": example["chosen"],
            "a2": example["rejected"],
            "rank": 1,
        }

    if {"response_1", "response_2", "preference"}.issubset(keys) and prompt_key is not None:
        pref = int(example["preference"])
        # HelpSteer2-Preference uses:
        # negative -> response_1 better; positive -> response_2 better; -100 invalid.
        if pref == -100 or pref == 0:
            return {
                "prompt": _get_prompt(),
                "a1": example["response_1"],
                "a2": example["response_2"],
                "rank": -1,
            }
        rank = 1 if pref < 0 else 0
        return {
            "prompt": _get_prompt(),
            "a1": example["response_1"],
            "a2": example["response_2"],
            "rank": rank,
        }

    if {"response_0", "response_1", "chosen"}.issubset(keys) and prompt_key is not None:
        chosen = int(example["chosen"])
        if chosen not in (0, 1):
            raise ValueError(f"chosen must be 0/1, got {example['chosen']}")
        return {
            "prompt": _get_prompt(),
            "a1": example["response_0"],
            "a2": example["response_1"],
            "rank": 1 if chosen == 0 else 0,
        }

    # Generic pairwise aliases
    if {"answer_0", "answer_1", "label"}.issubset(keys) and prompt_key is not None:
        label = int(example["label"])
        return {
            "prompt": _get_prompt(),
            "a1": example["answer_0"],
            "a2": example["answer_1"],
            "rank": 1 if label == 0 else 0,
        }

    if {"response_a", "response_b", "winner"}.issubset(keys) and prompt_key is not None:
        winner = str(example["winner"]).lower()
        if winner not in {"a", "b"}:
            return {"prompt": _get_prompt(), "a1": example["response_a"], "a2": example["response_b"], "rank": -1}
        return {
            "prompt": _get_prompt(),
            "a1": example["response_a"],
            "a2": example["response_b"],
            "rank": 1 if winner == "a" else 0,
        }

    raise ValueError(
        "Unsupported dataset schema. Expected one of: "
        "{prompt/a1/a2/rank}, {prompt/chosen/rejected}, {prompt/response_0/response_1/chosen}, "
        "{prompt/response_1/response_2/preference}. "
        f"Got keys={sorted(list(keys))}"
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
    def _safe_convert(x: dict[str, Any]) -> dict[str, Any]:
        try:
            out = convert_example_to_drpo_schema(x)
            out["prompt"] = _to_plain_text(out["prompt"])
            out["a1"] = _to_plain_text(out["a1"])
            out["a2"] = _to_plain_text(out["a2"])
            out["_valid"] = 1 if out["rank"] in [0, 1] else 0
            return out
        except Exception:
            return {"prompt": "", "a1": "", "a2": "", "rank": -1, "_valid": 0}

    standardized = ds.map(_safe_convert, remove_columns=remove_cols)
    standardized = standardized.filter(lambda x: x["_valid"] == 1)
    standardized = standardized.remove_columns(["_valid"])
    standardized = standardized.filter(lambda x: x["rank"] in [0, 1])
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


def _looks_like_scored_single_response_dataset(ds: Dataset) -> bool:
    cols = set(ds.column_names)
    score_cols = {"helpfulness", "correctness", "coherence", "complexity", "verbosity", "score"}
    return {"prompt", "response"}.issubset(cols) and len(cols.intersection(score_cols)) > 0


def build_pairwise_from_scored_responses(ds: Dataset) -> Dataset:
    rows_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ds:
        rows_by_prompt[row["prompt"]].append(row)

    score_cols = ["helpfulness", "correctness", "coherence", "complexity", "verbosity", "score"]
    pairs: list[dict[str, Any]] = []
    for prompt, rows in rows_by_prompt.items():
        if len(rows) < 2:
            continue

        def composite_score(r: dict[str, Any]) -> float:
            vals = []
            for c in score_cols:
                v = r.get(c, None)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if len(vals) == 0:
                return float("-inf")
            return sum(vals) / len(vals)

        best = max(rows, key=composite_score)
        worst = min(rows, key=composite_score)
        if best.get("response") == worst.get("response"):
            continue
        pairs.append(
            {
                "prompt": prompt,
                "a1": best["response"],
                "a2": worst["response"],
                "rank": 1,
            }
        )

    if len(pairs) == 0:
        raise ValueError("Failed to build pairwise data from scored dataset.")
    return Dataset.from_list(pairs)


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
        try:
            raw = load_dataset(dataset_name, dataset_config_name, split=dataset_split)
        except ValueError as e:
            # Some datasets expose only "default" config in specific mirrors/versions.
            # Fallback to default config to keep mixed training robust.
            if "BuilderConfig" in str(e) and "not found" in str(e):
                print(
                    f"[DRPO] dataset_config_name='{dataset_config_name}' not found for {dataset_name}. "
                    "Falling back to default config."
                )
                raw = load_dataset(dataset_name, split=dataset_split)
            else:
                raise
    else:
        raw = load_dataset(dataset_name, split=dataset_split)
    if _looks_like_scored_single_response_dataset(raw):
        print(f"[DRPO] Detected scored single-response schema for {dataset_name}; converting to pairwise.")
        raw = build_pairwise_from_scored_responses(raw)
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
