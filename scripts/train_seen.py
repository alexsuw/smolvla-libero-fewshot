"""Train the seen-domain policy. Leakage is checked before later allocation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from vla_fewshot.config import TrainConfig, load_config
from vla_fewshot.data.gates import maybe_assert_no_leakage
from vla_fewshot.data.layout import resolve_datasets_dir
from vla_fewshot.data.leakage import LeakageError
from vla_fewshot.logging.manifest import build_run_id
from vla_fewshot.storage.sync import execute_local_mirror
from vla_fewshot.training.compare import run_resume_compare_protocol
from vla_fewshot.training.full import require_full_training_runtime
from vla_fewshot.training.full_loop import prepare_full_training, run_full_training
from vla_fewshot.training.resume import (
    ResumeError,
    assert_override_allowlist,
    assert_resume_compatible,
)
from vla_fewshot.training.trainer import run_static_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/smoke.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/splits/target_splits.json"),
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--profile",
        choices=("static", "full"),
        default="full",
        help="static: CPU toy 200-step resume smoke. full: SmolVLA on CUDA.",
    )
    parser.add_argument(
        "--protocol",
        choices=("train", "resume-compare"),
        default="train",
        help="resume-compare runs 0→200 vs 0→100→200 with a fresh process.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        help="Allowed resume override: durable mirror destination.",
    )
    return parser


def _load_train_config(path: Path) -> TrainConfig:
    loaded = load_config(path)
    if not isinstance(loaded, TrainConfig):
        raise TypeError(f"{path} is not a train config")
    return loaded


def _full_run_dir(args: argparse.Namespace, config: TrainConfig, project_root: Path) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    from vla_fewshot.paths import resolve_paths

    try:
        paths = resolve_paths()
    except RuntimeError as error:
        raise SystemExit(
            f"{error}; set --output-dir or VLA_DATA_ROOT / VLA_RUNS_DIR. "
            "no GPU training was started."
        ) from error
    return paths.runs_dir / build_run_id(config, project_root=project_root)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")

    if args.profile == "full":
        try:
            require_full_training_runtime()
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 1

    try:
        maybe_assert_no_leakage(
            config_path=args.data_config,
            splits_path=args.split,
            output_root=args.output_root,
            stage="seen",
            required=args.profile == "full",
        )
    except LeakageError as error:
        print(str(error))
        return 1

    try:
        config = _load_train_config(args.config)
    except Exception as error:
        parser.error(f"invalid config: {error}")

    if args.resume_from is not None:
        try:
            config = assert_resume_compatible(args.resume_from, config)
        except ResumeError as error:
            print(str(error), file=sys.stderr)
            return 1

    assert_override_allowlist(
        {
            "log_freq": args.log_freq,
            "destination": args.destination,
            "stop_after": args.stop_after,
            "backup_dir": args.backup_dir,
            "output_dir": args.output_dir,
        }
    )
    project_root = Path.cwd()
    command = ["python", "scripts/train_seen.py", *sys.argv[1:]]

    if args.profile == "full":
        if args.protocol == "resume-compare":
            parser.error("resume-compare is the CPU static protocol; use --profile static")
        try:
            datasets_dir = resolve_datasets_dir(args.output_root)
            prepared = prepare_full_training(config, datasets_dir=datasets_dir)
            config = prepared["config"]
            run_dir = _full_run_dir(args, config, project_root)
            result = run_full_training(
                config=config,
                run_dir=run_dir,
                command=command,
                config_path=args.config,
                project_root=project_root,
                datasets_dir=datasets_dir,
                resume_from=args.resume_from,
                stop_after=args.stop_after,
                log_freq=args.log_freq,
                install_signal_handlers=True,
                run_id=run_dir.name,
                prepared=prepared,
            )
        except (RuntimeError, FileNotFoundError, FileExistsError, ResumeError) as error:
            print(str(error), file=sys.stderr)
            return 1
        mirror = args.backup_dir or args.destination
        if mirror is not None and result.status in {"completed", "stopped"}:
            execute_local_mirror(run_dir, mirror, execute=True)
        print(
            f"train status={result.status} step={result.global_step} "
            f"checkpoint={result.final_checkpoint}"
        )
        return 0 if result.status in {"completed", "stopped"} else 1

    if args.protocol == "resume-compare":
        if args.output_dir is None:
            parser.error("--output-dir is required for --protocol resume-compare")
        report = run_resume_compare_protocol(
            config=config,
            output_dir=args.output_dir,
            command=command,
            config_path=args.config,
            project_root=project_root,
            train_script=Path(__file__).resolve(),
            log_freq=args.log_freq,
            backup_dir=args.backup_dir or args.destination,
        )
        print(report["notes"])
        return 0 if report["passed"] else 1

    if args.output_dir is None:
        parser.error("--output-dir is required for static training")
    result = run_static_training(
        config=config,
        run_dir=args.output_dir,
        command=command,
        config_path=args.config,
        project_root=project_root,
        profile="static",
        resume_from=args.resume_from,
        stop_after=args.stop_after,
        log_freq=args.log_freq,
        install_signal_handlers=True,
        run_id=args.output_dir.name or build_run_id(config, project_root=project_root),
    )
    mirror = args.backup_dir or args.destination
    if mirror is not None and result.status in {"completed", "stopped"}:
        execute_local_mirror(args.output_dir, mirror, execute=True)
    return 0 if result.status in {"completed", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
