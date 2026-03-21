import os
import yaml
import torch

from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.utils import SIMPLE_CHAT_TEMPLATE

from trainer.drpo_utils import GPMwithRewardNetwork, estDPOStylePipeline, BTRewardNetwork
from trainer import DRPOConfig, DRPOTrainer


DATASETNAME = "Kyleyee/train_data_hh_for_drpo"
MODELNAME = "Kyleyee/Qwen2.5-1.5B-sft-hh"


def transform_dataset(dataset, seed=996):
    def process_split(split):
        original = dataset[split]

        swapped = original.map(lambda x: {
            "prompt": x["prompt"],
            "a1": x["a2"],
            "a2": x["a1"],
            "rank": 1 - x["rank"],
        })

        return concatenate_datasets([original, swapped]).shuffle(seed=seed)

    return DatasetDict({
        split: process_split(split)
        for split in dataset.keys()
    })


def main(script_args, training_args, model_args):
    ################
    # Model & Tokenizer
    ################
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
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )

    # Qwen2.5 常用设置
    tokenizer.eos_token = "<|im_end|>"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers()
            if buffer.dtype == torch.bool
        ]

    ################
    # Preference model
    ################
    if training_args.is_bt_model:
        if isinstance(training_args.preference_model_id, dict):
            preference_pipeline = estDPOStylePipeline(training_args.preference_model_id)
        else:
            preference_pipeline = BTRewardNetwork(
                training_args.preference_model_id,
                revision=training_args.preference_model_revision,
                pad_token_id=tokenizer.pad_token_id,
            )
    else:
        preference_pipeline = GPMwithRewardNetwork(training_args.preference_model_id)

    ################
    # Dataset
    ################
    if getattr(script_args, "dataset_config", None) and "revision" in script_args.dataset_config:
        dataset = load_dataset(
            script_args.dataset_name,
            revision=script_args.dataset_config["revision"],
        )
    else:
        dataset = load_dataset(script_args.dataset_name)

    dataset = transform_dataset(dataset)

    print(f"\033[32mLoaded dataset sample:\033[0m {dataset['train'][0]}")
    print(f"\033[32mLoaded swapped dataset sample:\033[0m {dataset['train'][-1]}")

    ################
    # Training
    ################
    trainer = DRPOTrainer(
        model=model,
        ref_model=ref_model,
        preference_model=preference_pipeline,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    script_args = ScriptArguments(
        dataset_name=DATASETNAME,
        dataset_train_split="train",
        dataset_test_split="test",
    )

    model_args = ModelConfig(
        model_name_or_path=MODELNAME,
    )

    with open("./examples/hh/config.yaml", "r") as f:
        training_args_config = yaml.safe_load(f)

    training_args = DRPOConfig(**training_args_config)

    main(script_args, training_args, model_args)