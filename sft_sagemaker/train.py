"""TRL SFT training entry point.

``train`` runs unchanged either in-process (``--local``) or inside a SageMaker
training job via the ``@remote`` decorator applied in ``launch.py``. Heavy
imports live inside the function so the launcher only needs the SageMaker SDK.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from sft_sagemaker.config import TrainConfig

logger = logging.getLogger(__name__)

MODEL_OUTPUT_S3_URI_ENV = "SFT_MODEL_OUTPUT_S3_URI"
HF_TOKEN_SSM_PARAMETER_ENV = "SFT_HF_TOKEN_SSM_PARAMETER"

# The SageMaker ML storage volume (sized by ``volume_size``) backs /tmp and
# /opt/ml; the root filesystem is small, so model downloads must not land in ~.
SAGEMAKER_SCRATCH_DIR = "/tmp/sft"


def running_on_sagemaker() -> bool:
    return "TRAINING_JOB_NAME" in os.environ


def train(cfg: TrainConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_name = cfg.run_name or os.environ.get("TRAINING_JOB_NAME") or time.strftime("sft-%Y%m%d-%H%M%S")
    if running_on_sagemaker():
        os.environ.setdefault("HF_HOME", f"{SAGEMAKER_SCRATCH_DIR}/hf-home")
        output_dir = Path(cfg.output_dir or f"{SAGEMAKER_SCRATCH_DIR}/output/{run_name}")
    else:
        output_dir = Path(cfg.output_dir or f"outputs/{run_name}")
    _load_hf_token()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    use_bf16 = cfg.bf16 if cfg.bf16 is not None else torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    train_dataset = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_train_split)
    if cfg.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(cfg.max_train_samples, len(train_dataset))))

    eval_dataset = None
    if cfg.dataset_eval_split:
        eval_dataset = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_eval_split)
        if cfg.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(cfg.max_eval_samples, len(eval_dataset))))

    peft_config = None
    if cfg.use_peft:
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules
            if cfg.lora_target_modules == "all-linear"
            else [m.strip() for m in cfg.lora_target_modules.split(",")],
            task_type="CAUSAL_LM",
        )

    args = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        max_length=cfg.max_length,
        packing=cfg.packing,
        assistant_only_loss=cfg.assistant_only_loss,
        dataset_text_field=cfg.dataset_text_field,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=use_bf16,
        seed=cfg.seed,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=cfg.eval_steps,
        save_strategy="no",
        report_to=cfg.report_to,
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id,
        model_init_kwargs={"dtype": torch.bfloat16 if use_bf16 else "auto"},
    )

    trainer = SFTTrainer(
        model=cfg.model_name_or_path,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    metrics: dict[str, Any] = dict(train_result.metrics)
    if eval_dataset is not None:
        metrics.update(trainer.evaluate())

    trainer.save_model(str(output_dir))
    trainer.save_state()

    result: dict[str, Any] = {
        "run_name": run_name,
        "metrics": metrics,
        "log_history": trainer.state.log_history,
        "output_dir": str(output_dir),
        "model_s3_uri": None,
        "hub_model_id": None,
    }

    if trainer.is_world_process_zero():
        if cfg.push_to_hub:
            trainer.push_to_hub()
            result["hub_model_id"] = trainer.hub_model_id
        if cfg.upload_to_s3 and os.environ.get(MODEL_OUTPUT_S3_URI_ENV):
            result["model_s3_uri"] = upload_dir_to_s3(
                output_dir, f"{os.environ[MODEL_OUTPUT_S3_URI_ENV].rstrip('/')}/{run_name}"
            )

    return result


def _load_hf_token() -> None:
    parameter_name = os.environ.get(HF_TOKEN_SSM_PARAMETER_ENV)
    if os.environ.get("HF_TOKEN") or not parameter_name:
        return

    import boto3

    response = boto3.client("ssm").get_parameter(Name=parameter_name, WithDecryption=True)
    os.environ["HF_TOKEN"] = response["Parameter"]["Value"]
    logger.info("Loaded Hugging Face token from SSM parameter %s", parameter_name)


def upload_dir_to_s3(local_dir: Path, s3_uri: str) -> str:
    import boto3

    bucket, _, prefix = s3_uri.removeprefix("s3://").partition("/")
    s3 = boto3.client("s3")
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    for path in files:
        key = f"{prefix}/{path.relative_to(local_dir).as_posix()}"
        s3.upload_file(str(path), bucket, key)
    logger.info("Uploaded %d files to %s", len(files), s3_uri)
    return s3_uri
