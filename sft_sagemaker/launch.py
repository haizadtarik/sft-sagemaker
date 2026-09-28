"""Launch SFT training locally or as a SageMaker training job via ``@remote``.

Examples:
    python -m sft_sagemaker.launch --max-steps 100
    python -m sft_sagemaker.launch --use-peft --learning-rate 2e-4 --instance-type ml.g5.12xlarge \
        --torchrun --nproc-per-node 4
    python -m sft_sagemaker.launch --local --model-name-or-path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
        --max-steps 2 --max-train-samples 16
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from sft_sagemaker.config import TrainConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SAGEMAKER_CONFIG_PATH = REPO_ROOT / ".sagemaker" / "config.yaml"
TRAIN_REQUIREMENTS_PATH = REPO_ROOT / "requirements-train.txt"
PACKAGE_DIR_NAME = "sft_sagemaker"

DEFAULT_INSTANCE_TYPE = "ml.g5.2xlarge"
# Must be a PyTorch training DLC whose Python matches the local interpreter:
# the @remote decorator unpickles the function with the job's Python.
DEFAULT_PYTORCH_VERSION = "2.8.0"


def main(argv: list[str] | None = None) -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args(argv)
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig)})

    if args.local:
        from sft_sagemaker.train import train

        result = train(cfg)
    else:
        result = run_remote(cfg, args)

    print(json.dumps({k: v for k, v in result.items() if k != "log_history"}, indent=2, default=str))


def run_remote(cfg: TrainConfig, args: argparse.Namespace) -> dict[str, Any]:
    if not SAGEMAKER_CONFIG_PATH.exists():
        sys.exit(f"{SAGEMAKER_CONFIG_PATH} not found. Provision the infrastructure first with `make infra`.")
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = str(SAGEMAKER_CONFIG_PATH)
    os.environ.setdefault("SAGEMAKER_SUPPRESS_V2_WARNING", "1")

    import boto3
    import sagemaker
    from sagemaker.remote_function import remote

    from sft_sagemaker.train import train

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    # include_local_workdir packages os.getcwd(), so anchor it at the repo root.
    os.chdir(REPO_ROOT)

    remote_train = remote(
        train,
        sagemaker_session=session,
        image_uri=args.image_uri or default_image_uri(session.boto_region_name, args.instance_type),
        instance_type=args.instance_type,
        volume_size=args.volume_size,
        max_runtime_in_seconds=int(args.max_runtime_hours * 3600),
        keep_alive_period_in_seconds=args.keep_alive_seconds,
        job_name_prefix=args.job_name_prefix,
        dependencies=str(TRAIN_REQUIREMENTS_PATH),
        include_local_workdir=True,
        custom_file_filter=package_only_filter,
        use_torchrun=args.torchrun,
        nproc_per_node=args.nproc_per_node,
    )
    return remote_train(cfg)


def default_image_uri(region: str, instance_type: str) -> str:
    from sagemaker import image_uris

    py_version = f"py{sys.version_info.major}{sys.version_info.minor}"
    try:
        return image_uris.retrieve(
            framework="pytorch",
            region=region,
            version=DEFAULT_PYTORCH_VERSION,
            py_version=py_version,
            instance_type=instance_type,
            image_scope="training",
        )
    except ValueError as err:
        sys.exit(
            f"No PyTorch {DEFAULT_PYTORCH_VERSION} training image for {py_version}: {err}\n"
            "Run the launcher with Python 3.12, or pass --image-uri with an image matching your Python version."
        )


def package_only_filter(path: str, names: list[str]) -> list[str]:
    """shutil.copytree ignore callback: ship only the package's .py files.

    Keeps .env, Terraform state, virtualenvs and local outputs out of the job.
    """
    if Path(path).resolve() == REPO_ROOT:
        return [n for n in names if n != PACKAGE_DIR_NAME]
    return [
        n
        for n in names
        if n == "__pycache__" or (not n.endswith(".py") and not os.path.isdir(os.path.join(path, n)))
    ]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    launch = parser.add_argument_group("launch")
    launch.add_argument("--local", action="store_true", help="Run in this process instead of on SageMaker.")
    launch.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    launch.add_argument("--image-uri", help="Override the PyTorch training image.")
    launch.add_argument("--volume-size", type=int, default=100, help="Training volume size in GB.")
    launch.add_argument("--max-runtime-hours", type=float, default=24)
    launch.add_argument(
        "--keep-alive-seconds",
        type=int,
        default=0,
        help="Keep the instance in a warm pool between jobs to skip provisioning (billed while idle).",
    )
    launch.add_argument("--job-name-prefix", default="sft")
    launch.add_argument("--torchrun", action="store_true", help="Launch with torchrun for multi-GPU instances.")
    launch.add_argument("--nproc-per-node", type=int, help="Processes per node with --torchrun (default: all GPUs).")

    training = parser.add_argument_group("training (see sft_sagemaker/config.py)")
    for f in dataclasses.fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        default = f.default
        type_name = str(f.type)
        if "bool" in type_name:
            training.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=default)
        else:
            caster = int if type_name.startswith("int") else float if type_name.startswith("float") else str
            training.add_argument(flag, dest=f.name, type=caster, default=default, help=f"default: {default}")

    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
