from pathlib import Path

from vla_fewshot.evaluation.cli import eval_cell_output_dir


def test_run_dir_keeps_step_prefix_for_a_single_checkpoint(tmp_path: Path) -> None:
    output = eval_cell_output_dir(
        tmp_path,
        label="step_100000",
        task="black_bowl_plate",
        run_dir=True,
        n_tasks=3,
    )
    assert output == tmp_path / "step_100000" / "black_bowl_plate"


def test_single_checkpoint_without_run_dir_uses_task_subdir(tmp_path: Path) -> None:
    output = eval_cell_output_dir(
        tmp_path,
        label="ckpt",
        task="black_bowl_plate",
        run_dir=False,
        n_tasks=3,
    )
    assert output == tmp_path / "black_bowl_plate"
