**结构：**
```
.
└── DRPO4LLM/
    ├── examples/
    │   └── tldr/
    │       ├── drpo.py
    │       └── config.yaml
    ├── trainer/
    │   ├── __init__.py
    │   ├── drpo_config.py
    │   ├── drpo_trainer.py
    │   └── drpo_utils.py
    ├── requirements.txt
    ├── test.py
    └── introduction.md
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
    > `pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121`
    > `pip install -r requirements.txt`
3. 训练模型
    > 在主目录下`accelerate launch -m examples.tldr.drpo`