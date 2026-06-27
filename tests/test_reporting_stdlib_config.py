"""Tests for stdlib-only experiment reporting config reads."""

from __future__ import annotations

from pathlib import Path

from scripts.reporting.config_projection import project_params
from scripts.reporting.run_reader import load_run_config
from scripts.reporting.schema import ParamAlias
from scripts.reporting.summarize import build_parser, summarize_from_args


def test_project_params_reads_list_values_without_omegaconf(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        run_name="run_a",
        dice_3d=0.31,
        filters=[16, 32, 64],
        kernel_size=[3, 3, 3],
    )

    cfg = load_run_config(run_dir)
    params = project_params(
        cfg,
        [
            ParamAlias(alias="Filters", config_path="model.filters"),
            ParamAlias(alias="Kernel Size", config_path="model.kernel_size"),
        ],
    )

    assert params == {
        "Filters": "[16, 32, 64]",
        "Kernel Size": "[3, 3, 3]",
    }


def test_summarize_sorts_flat_table_by_3d_dice(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        run_name="lower_dice",
        dice_3d=0.42,
        filters=[16, 32, 64],
        kernel_size=[3, 3, 3],
    )
    _write_run(
        tmp_path,
        run_name="higher_dice",
        dice_3d=0.61,
        filters=[32, 64, 128],
        kernel_size=[5, 5, 5],
    )

    parser = build_parser()
    raw_args = [
        "--experiment-dir",
        str(tmp_path),
        "--primary-metric",
        "dice_3d",
        "--primary-direction",
        "max",
        "--metrics",
        "dice_3d",
        "--param",
        "Filters=model.filters",
        "--param",
        "Kernel Size=model.kernel_size",
        "--columns",
        "run_name,Filters,Kernel Size,dice_3d",
        "--sort-metric",
        "dice_3d:desc",
        "--format",
        "csv",
    ]
    args = parser.parse_args(raw_args)

    output = summarize_from_args(args, argv=raw_args)

    assert output.index("higher_dice") < output.index("lower_dice")
    assert '"[32, 64, 128]"' in output
    assert '"[5, 5, 5]"' in output


def _write_run(
    experiment_dir: Path,
    run_name: str,
    dice_3d: float,
    filters: list[int],
    kernel_size: list[int],
) -> Path:
    run_dir = experiment_dir / run_name
    hydra_dir = run_dir / ".hydra"
    best_dir = run_dir / "models" / "best"
    hydra_dir.mkdir(parents=True)
    best_dir.mkdir(parents=True)

    filters_text = "[" + ", ".join(str(value) for value in filters) + "]"
    kernel_text = "\n".join(f"    - {value}" for value in kernel_size)
    (hydra_dir / "config.yaml").write_text(
        "\n".join(
            [
                "model:",
                "  name: DynUNet",
                f"  filters: {filters_text}",
                "  kernel_size:",
                kernel_text,
                "training:",
                "  amp: true",
                "  max_steps: 100000",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (best_dir / "best_model_step_100_metrics.csv").write_text(
        "\n".join(
            [
                "metric_key,metric_value",
                f"dice_3d,{dice_3d}",
                "hd95_3d,12.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return run_dir
