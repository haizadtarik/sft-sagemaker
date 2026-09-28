from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Hyperparameters for a single SFT run.

    Instances are pickled by the SageMaker ``@remote`` decorator and shipped to
    the training job, so keep this module free of heavy imports.
    """

    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    dataset_name: str = "trl-lib/Capybara"
    dataset_config: str | None = None
    dataset_train_split: str = "train"
    dataset_eval_split: str | None = None
    dataset_text_field: str = "text"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_steps: float = 0.03
    """Integer step count, or a float in [0, 1) as a fraction of total steps."""
    weight_decay: float = 0.0
    max_length: int = 1024
    packing: bool = False
    assistant_only_loss: bool = False
    gradient_checkpointing: bool = True
    bf16: bool | None = None
    seed: int = 42

    use_peft: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    """``all-linear`` or a comma-separated list of module names."""

    logging_steps: int = 10
    eval_steps: int | None = None
    report_to: str = "none"

    run_name: str | None = None
    output_dir: str | None = None
    upload_to_s3: bool = True
    push_to_hub: bool = False
    hub_model_id: str | None = None
