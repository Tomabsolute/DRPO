**结构：**
```
.
└── DRPO4LLM/
    ├── examples/
    │   └── tldr/
    │       ├── drpo.py
    │       └── config.yaml
    │   └── hh/
    │       ├── drpo.py
    │       └── config.yaml
    ├── trainer/
    │   ├── __init__.py
    │   ├── drpo_config.py
    │   ├── drpo_trainer.py
    │   └── drpo_utils.py
    ├── trainer_2/
    │   ├── __init__.py
    │   ├── drpo_config.py
    │   ├── drpo_trainer.py
    │   └── drpo_utils.py
    ├── requirements.txt
    ├── compare_tldr.py
    ├── compare_hh.py
    └── readme.md
```
**accelertate配置**
```
compute_environment: LOCAL_MACHINE
debug: true
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
**如何使用：**
1. 创建conda环境，不细说
2. 下载软件包
    > `pip install -r requirements.txt`
3. 训练模型
    > 在主目录下`accelerate launch -m examples.tldr.drpo`

`compare_tldr.py`的运行方式：
```
python compare_tldr.py \
  --base_model cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
  --drpo_model output/tldr \
  --dataset_name trl-lib/tldr \
  --split test \
  --num_samples 50 \
  --output_file results1/compare_tldr_outputs.jsonl
```

`compare_hh.py`的运行方式：
```
python compare_hh_outputs.py \
  --base_model Kyleyee/Qwen2.5-1.5B-sft-hh \
  --drpo_model ./output/hh/ \
  --dataset_name Dahoas/full-hh-rlhf \
  --split test \
  --num_samples 50 \
  --output_file results/compare_hh_outputs.jsonl
```

`accelerate launch -m examples.general.drpo --config examples/general/config.yaml`

```
accelerate launch -m examples.general.drpo \
  --config examples/general/config.yaml \
  2>&1 | tee train.log

tail -n 100 train.log

grep -i error train.log
```