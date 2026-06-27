from mss.cli import main

from tests.conftest import write_minimal_raw


def test_cli_run_ingest_with_absolute_paths(tmp_path) -> None:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    write_minimal_raw(raw)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    cfg = configs_dir / "anyrun.toml"
    cfg.write_text(
        f'[paths]\nraw = "{raw.as_posix()}"\n'
        f'interim = "{interim.as_posix()}"\n'
        f'processed = "{processed.as_posix()}"\n',
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            "--configs-dir",
            str(configs_dir),
            "--run",
            "anyrun",
            "--pipeline",
            "data.prepare",
            "--skip-validate",
        ]
    )
    assert code == 0
    assert (interim / "ingest_manifest.json").is_file()
    assert (interim / "returns_panel.parquet").is_file()
    assert (interim / "targets.parquet").is_file()
    assert (interim / "targets_manifest.json").is_file()


def test_cli_run_all_pipelines_omitting_pipeline_flag(tmp_path) -> None:
    """Full stack through dataset.package (excludes model.train; needs torch-geometric extras)."""
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    write_minimal_raw(raw)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    cfg = configs_dir / "def.toml"
    cfg.write_text(
        f'[paths]\nraw = "{raw.as_posix()}"\n'
        f'interim = "{interim.as_posix()}"\n'
        f'processed = "{processed.as_posix()}"\n'
        "\n[graph]\nret_col = \"ret_used\"\nalign_feature_dates_with_targets = true\n\n"
        "[graph.rolling_window]\nlength = 1\nmin_obs_frac = 1.0\n\n"
        "[graph.universe]\nuniverse_mode = \"fixed_replace\"\nn_nodes = 1\n"
        "selection_rule = \"mcap\"\nrebalance_freq = \"monthly\"\nn_jobs = 1\nprogress_every = 10\n\n"
        "[graph.edges]\ndependence = \"spearman\"\ntop_k = 1\nsymmetrize = true\n"
        "n_jobs = 1\nupdate_every = 1\n\n"
        "[graph.node_features]\nrolling_mean = true\nrolling_vol = true\n"
        "n_jobs = 1\nprogress_every = 10\n\n"
        "[graph.output]\ntarget_column = \"log_rv_fwd_30\"\n\n"
        "[dataset]\ntrain_end = \"2020-01-02\"\nval_end = \"2020-01-03\"\n"
        "test_end = \"2025-12-31\"\nscaler_type = \"standard\"\n"
        'feature_columns = ["rolling_mean", "rolling_vol"]\n'
        "write_scaled_features = true\n",
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            "--configs-dir",
            str(configs_dir),
            "--run",
            "def",
            "--skip-validate",
            "--pipeline",
            "data.prepare",
            "--pipeline",
            "graph.prepare",
            "--pipeline",
            "dataset.package",
        ]
    )
    assert code == 0
    assert (interim / "ingest_manifest.json").is_file()
    assert (interim / "returns_panel.parquet").is_file()
    assert (interim / "targets.parquet").is_file()
    assert (interim / "targets_manifest.json").is_file()
    g = interim / "graphs"
    assert (g / "universe.parquet").is_file()
    assert (g / "node_features.parquet").is_file()
    assert (g / "edges.parquet").is_file()
    ds = interim / "dataset"
    assert (ds / "splits.parquet").is_file()
    assert (ds / "scaler_params.json").is_file()
    assert (ds / "manifest.json").is_file()
    assert (ds / "node_features_scaled.parquet").is_file()
