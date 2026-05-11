from dataclasses import dataclass, field
from typing import Optional, Union, Callable

from transformers import TrainingArguments

@dataclass
class DRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`OnlineDRPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse]
    """

    # 学习率重写，保证训练的稳定性
    learning_rate: float = field(
        default=5e-7,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of"
            "transformers.TrainingArguments."
        }
    )
    
    # 检测控制模型和偏好模型是否为同一个模型
    model_and_preference_share_basemodel: bool = field(
        default=False,
        metadata={"help": "Whether the model and preference model share the same base model (e.g. both from Qwen2.5 or Pythia...)"},
    )

    # 偏好模型的路径
    preference_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the preference model."
        }
    )

    # 偏好模型的额外参数
    preference_model_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Additional arguments for the preference model, e.g. `{'indifferent': False, 'random': False, 'reverse': False}`."
        }
    )

    # 最长生成字节数
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    
    # 提示+回答的最大长度
    max_length: int = field(
        default=256,
        metadata={
            "help": "Maximum total length of the sequence (prompt + completion) used to compute log probabilities. If "
            "the sequence exceeds this limit, the leftmost tokens will be truncated to preserve as much of the "
            "completion as possible."
        },
    )
    
    # 生成过程中采样与策略模型的温度，越高越随机
    generate_temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling and policy model. The higher the temperature, the more random the completions."},
    )
    
    # 未能生成EOS结束符的惩罚值
    missing_eos_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Penalty applied to the score when the model fails to generate an EOS token. This is useful to "
            "encourage to generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be "
            "a positive value."
        },
    )
    
    # 控制偏差的参数，beta越高偏差越小
    beta: list[float] = field(
        default_factory=lambda: [0.1],
        metadata={
            "help": "Parameter controlling the deviation from the reference model. Higher β means less deviation from "
            "the reference model. For the IPO loss (`loss_type='ipo'`), β is the regularization parameter denoted by "
            "τ in the [paper](https://huggingface.co/papers/2310.12036). If a list of floats is provided then the β "
            "is selected for each new epoch and the last β is used for the rest of the epochs."
        },
    )

    # 定义了处理数据集时使用的进程数
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    
    # 标志位表示是否在模型训练过程中禁用 dropout
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
    )

    # 标志位应用在 DeepSpeed ZeRO-3 配置中，表示是否在生成阶段收集模型权重，以提高生成速度。
    ds_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,"
            "improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU,"
            "albeit at the cost of slower generation."
        },
    )

    # 定义了与参考模型比较的新生成回复的数量
    num_astar: int = field(
        default=1,
        metadata={
            "help": "Number of newly generated completions to compare with the reference model."
        }
    )
    
    # 是一个可选的工具列表（可调用函数），这些工具对模型可见，但如果模板不支持调用函数，则此参数无效。
    tools: Optional[list[Union[dict, Callable]]] = field(
        default=None,
        metadata={
            "help": "List of tools (callable functions) that will be accessible to the model. If the template does "
            "not support function calling, this argument will have no effect."
        },
    )

    # 提示的最大长度
    max_prompt_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum length of the prompt."},
    )

    # 回复的最大长度
    max_completion_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum length of the completion."},
    )

    # 标志位表示是否预计算数据集的偏好评分，以提高训练效率
    precompute_preference_score: bool = field(
        default=False,
        metadata={"help": "Whether to precompute the preference score for the dataset."},
    )

    # 定义了每隔多少步记录一次训练损失
    logging_steps: int = field(
        default=10,
        metadata={"help": "Number of steps to log the training loss."},
    )

    # 定义了训练的周期数
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Number of training epochs."},
    )

    # 偏好模型是否使用了BT framework
    is_bt_model: bool = field(
        default=True,
        metadata={"help": "Whether the preference model uses BT framework."},
    )

    # 偏好模型的ID
    preference_model_id: Optional[str] = field(
        default="siebert/sentiment-roberta-large-english",
        metadata={"help": "Model ID of the preference model."},
    )

    # 偏好模型的版本
    preference_model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision of the preference model."},
    )

    # 处理重要性采样的方法
    ratio_processing: Union[str, None] = field(
        default=None,
        metadata={"help": "Processing method for the IS estimator. "
        "`clip` uses Clip-DRPO style correction clipping (Eq. 4.3), "
        "`self_normalize` uses self-normalized ratio, and `None` uses raw ratio."},
    )

    # 重要性采样比的上限
    clipbound: Optional[float] = field(
        default=10.0,
        metadata={"help": "Deprecated by Clip-DRPO correction clipping; kept for backward compatibility."},
    )

    # clip 路径选择:
    # new  -> Clip-DRPO correction clipping (当前 trainer 实现)
    # orig -> 原版 ratio clamp: clamp(r, 1/clipbound, clipbound)
    clip_mode: str = field(
        default="new",
        metadata={"help": "Clip mode when ratio_processing='clip'. Choose from: 'new' or 'orig'."},
    )

    # 控制参考模型前向传递的温度
    forward_temperature: Optional[float] = field(
        default=0.9,
        metadata={"help": "Temperature for the forward pass of ref_model."},
    )

    # 是否使用重要性采样估计器
    loss1_only: bool = field(
        default=False,
        metadata={"help": "Whether to only use the Importance Sampling estimator"},
    )
    
    # 是否使用直接方法估计器
    loss2_only: bool = field(
        default=False,
        metadata={"help": "Whether to only use the Direct Method estimator"},
    )

    # 是否在生成回复后添加EOS结束符
    eos_after_completion: bool = field(
        default=False,
        metadata={"help": "Whether to add eos token after the completion., choose False when your data applies chat template, choose True when your data is not and you want your generation contain the eos"},
    )

    # 初始化配置在类实例化后，检查 beta 参数是否为长度为 1 的列表，如果是，则将其转换为单个浮点数。
    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.beta, "__len__") and len(self.beta) == 1:
            self.beta = self.beta[0]
        if self.clip_mode not in {"new", "orig"}:
            raise ValueError(f"clip_mode must be one of ['new', 'orig'], got: {self.clip_mode}")
