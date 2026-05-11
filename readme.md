**结构：**
```
.
└── DRPO4LLM/
    ├── examples/
    │   └── general/
    │       ├── drpo.py
    │       ├── test_model.py
    │       ├── config_clip.yaml
    │       └── config_clip_orig.yaml
    ├── model
    ├── dataset
    ├── output   
    ├── trainer/
    │   ├── __init__.py
    │   ├── drpo_config.py
    │   ├── drpo_trainer.py
    │   └── drpo_utils.py
    ├── 1.png
    ├── 2.png
    ├── draw.py
    ├── hfd.sh
    ├── requirements.txt
    ├── compare_tldr.py
    ├── compare_hh.py
    ├── test.py
    └── readme.md
```
**accelertate配置**
```
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
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
    > 在主目录下`accelerate launch -m examples.general.drpo --config examples/general/config_*.yaml`



**clip**
```
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
accelerate launch -m examples.general.drpo \
  --config examples/general/config_clip.yaml \
  2>&1 | tee train.log

tail -n 100 train.log

grep -i error train.log
```

**clip_orig**
```
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
accelerate launch -m examples.general.drpo \
  --config examples/general/config_clip_orig.yaml \
  2>&1 | tee train_orig.log

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
accelerate launch -m examples.general.drpo \
  --config examples/general/config_clip_orig.yaml \
  --resume_from_checkpoint ./output/general_clip_orig/checkpoint-900 \
  2>&1 | tee train_orig_resume.log

  grep -i error train_orig.log
```
