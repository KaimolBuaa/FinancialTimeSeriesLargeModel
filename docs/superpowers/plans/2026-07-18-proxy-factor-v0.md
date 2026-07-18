# ProxyFactor-v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and verify a complete 128-factor, all-A-share, daily ProxyFactor-v0 dataset for 2008–2025 from the local Qlib K-line corpus.

**Architecture:** A fixed factor registry defines 16 one-day formulas and 14 rolling families over eight windows. A lazy Qlib adapter queries the dynamic `all` universe in 32-factor shards, a yearly materializer joins and atomically publishes factor/label Parquet partitions, and independent quality and store modules verify and read the result without loading the full corpus.

**Tech Stack:** Python 3.12, PyQlib 0.9.7, pandas 2.3, PyArrow 24, Parquet/Zstandard, `unittest`, SHA-256 manifests.

---

## File Map

- Create `src/factorpanel_data/proxy_registry.py`: immutable 128-factor and 3-label formula registry.
- Create `src/factorpanel_data/proxy_config.py`: validated generation configuration and config fingerprint.
- Create `src/factorpanel_data/qlib_proxy.py`: lazy PyQlib initialization and shard queries.
- Create `src/factorpanel_data/proxy_materialize.py`: yearly generation, atomic publication, resume state and manifest assembly.
- Create `src/factorpanel_data/proxy_quality.py`: partition/global statistics, causality/boundary checks and final verification.
- Create `src/factorpanel_data/proxy_store.py`: column-projected reads for one factor/date range and at most 512 assets.
- Create `src/factorpanel_data/proxy_cli.py`: `generate`, `verify`, `status` and `sample` commands.
- Create `configs/proxy_factor_v0.json`: authoritative 2008–2025 generation configuration.
- Modify `pyproject.toml`: add a `proxy` optional dependency and `factorpanel-proxy` console script.
- Modify `src/factorpanel_data/__init__.py`: export stable registry/config/store APIs.
- Create `tests/test_proxy_registry.py`, `tests/test_proxy_qlib.py`, `tests/test_proxy_materialize.py`, `tests/test_proxy_quality.py`, `tests/test_proxy_store.py`, and `tests/test_proxy_cli.py`.
- Generate ignored artifacts under `resources/data/proxy_factor_v0/`.

## Task 1: Fixed 128-Factor Registry

**Files:**
- Create: `src/factorpanel_data/proxy_registry.py`
- Create: `tests/test_proxy_registry.py`
- Modify: `src/factorpanel_data/__init__.py`

- [ ] **Step 1: Write the failing registry tests**

```python
from factorpanel_data.proxy_registry import (
    FACTOR_WINDOWS,
    build_label_registry,
    build_proxy_factor_registry,
)


def test_registry_has_exact_stable_contract():
    factors = build_proxy_factor_registry()
    assert FACTOR_WINDOWS == (2, 3, 5, 10, 20, 30, 60, 120)
    assert len(factors) == 128
    assert len({item.name for item in factors}) == 128
    assert [item.name for item in factors[:3]] == ["pf_kmid", "pf_klen", "pf_kmid2"]
    assert factors[-1].name == "pf_vstd_120"
    assert all(",-" not in item.expression.replace(" ", "") for item in factors)


def test_labels_are_separate_and_forward_looking():
    labels = build_label_registry()
    assert [item.name for item in labels] == ["ret_1d", "ret_5d", "ret_20d"]
    assert [item.horizon for item in labels] == [1, 5, 20]
    assert all("Ref($close,-" in item.expression.replace(" ", "") for item in labels)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=src resources/.venv/bin/python -m unittest tests.test_proxy_registry -v
```

Expected: import failure for `factorpanel_data.proxy_registry`.

- [ ] **Step 3: Implement immutable definitions and builders**

```python
from dataclasses import dataclass

FACTOR_WINDOWS = (2, 3, 5, 10, 20, 30, 60, 120)


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    expression: str
    family: str
    window: int | None = None


@dataclass(frozen=True)
class LabelDefinition:
    name: str
    expression: str
    horizon: int


def build_proxy_factor_registry() -> tuple[FactorDefinition, ...]:
    base = (
        FactorDefinition("pf_kmid", "($close-$open)/($open+1e-12)", "kbar"),
        FactorDefinition("pf_klen", "($high-$low)/($open+1e-12)", "kbar"),
        FactorDefinition("pf_kmid2", "($close-$open)/($high-$low+1e-12)", "kbar"),
        FactorDefinition("pf_kup", "($high-Greater($open,$close))/($open+1e-12)", "kbar"),
        FactorDefinition("pf_kup2", "($high-Greater($open,$close))/($high-$low+1e-12)", "kbar"),
        FactorDefinition("pf_klow", "(Less($open,$close)-$low)/($open+1e-12)", "kbar"),
        FactorDefinition("pf_klow2", "(Less($open,$close)-$low)/($high-$low+1e-12)", "kbar"),
        FactorDefinition("pf_ksft", "(2*$close-$high-$low)/($open+1e-12)", "kbar"),
        FactorDefinition("pf_ksft2", "(2*$close-$high-$low)/($high-$low+1e-12)", "kbar"),
        FactorDefinition("pf_open_close", "$open/($close+1e-12)-1", "price_ratio"),
        FactorDefinition("pf_high_close", "$high/($close+1e-12)-1", "price_ratio"),
        FactorDefinition("pf_low_close", "$low/($close+1e-12)-1", "price_ratio"),
        FactorDefinition("pf_vwap_close", "$vwap/($close+1e-12)-1", "price_ratio"),
        FactorDefinition("pf_return_1", "$close/(Ref($close,1)+1e-12)-1", "change"),
        FactorDefinition("pf_volume_change_1", "$volume/(Ref($volume,1)+1e-12)-1", "change"),
        FactorDefinition("pf_amount_change_1", "$amount/(Ref($amount,1)+1e-12)-1", "change"),
    )
    templates = {
        "roc": "$close/(Ref($close,{w})+1e-12)-1",
        "ma": "$close/(Mean($close,{w})+1e-12)-1",
        "std": "Std($close,{w})/($close+1e-12)",
        "beta": "Slope($close,{w})/($close+1e-12)",
        "rsqr": "Rsquare($close,{w})",
        "max": "$close/(Max($high,{w})+1e-12)-1",
        "min": "$close/(Min($low,{w})+1e-12)-1",
        "rsv": "($close-Min($low,{w}))/(Max($high,{w})-Min($low,{w})+1e-12)",
        "corr": "Corr($close,Log($volume+1),{w})",
        "cord": "Corr($close/(Ref($close,1)+1e-12)-1,Log($volume/(Ref($volume,1)+1e-12)+1),{w})",
        "cntd": "Mean($close>Ref($close,1),{w})-Mean($close<Ref($close,1),{w})",
        "sumd": "(Sum(Greater($close-Ref($close,1),0),{w})-Sum(Greater(Ref($close,1)-$close,0),{w}))/(Sum(Abs($close-Ref($close,1)),{w})+1e-12)",
        "vma": "$volume/(Mean($volume,{w})+1e-12)-1",
        "vstd": "Std($volume,{w})/(Mean($volume,{w})+1e-12)",
    }
    rolling = tuple(
        FactorDefinition(
            f"pf_{family}_{window}",
            template.format(w=max(window, 3) if family == "rsqr" else window),
            family,
            window,
        )
        for family, template in templates.items()
        for window in FACTOR_WINDOWS
    )
    result = base + rolling
    if len(result) != 128 or len({item.name for item in result}) != 128:
        raise RuntimeError("ProxyFactor-v0 registry must contain 128 unique factors")
    return result


def build_label_registry() -> tuple[LabelDefinition, ...]:
    return tuple(
        LabelDefinition(
            name=f"ret_{horizon}d",
            expression=f"Log(Ref($close,-{horizon})/($close+1e-12))",
            horizon=horizon,
        )
        for horizon in (1, 5, 20)
    )
```

`pf_rsqr_2` 必须使用 `Rsquare($close,3)`：两点回归的 R² 恒为 1，会触发 99% 近常数质量门槛。名称和名义窗口仍保持为 2，manifest 以实际表达式为准。

- [ ] **Step 4: Run registry tests and full regression tests**

```bash
PYTHONPATH=src resources/.venv/bin/python -m unittest tests.test_proxy_registry -v
PYTHONPATH=src /opt/miniconda3/envs/factorpanel-fm/bin/python -m unittest discover -s tests
```

Expected: registry tests pass and the existing 125 tests remain green.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/factorpanel_data/proxy_registry.py src/factorpanel_data/__init__.py tests/test_proxy_registry.py
git commit -m "feat: define ProxyFactor v0 registry"
```

## Task 2: Configuration and Qlib Query Adapter

**Files:**
- Create: `src/factorpanel_data/proxy_config.py`
- Create: `src/factorpanel_data/qlib_proxy.py`
- Create: `configs/proxy_factor_v0.json`
- Create: `tests/test_proxy_qlib.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing config and fake-provider tests**

```python
def test_default_config_covers_full_requested_scope(tmp_path):
    config = ProxyFactorConfig(
        provider_uri=tmp_path / "qlib",
        output_root=tmp_path / "proxy",
        start_year=2008,
        end_year=2025,
        universe="all",
        factor_shard_size=32,
    )
    assert config.years == tuple(range(2008, 2026))
    assert config.factor_shard_size == 32
    assert len(config.fingerprint) == 64


def test_query_year_joins_shards_and_trims_to_target_year(fake_provider):
    frame = query_factor_year(fake_provider, year=2008, shard_size=32)
    assert frame.index.names == ["instrument", "datetime"]
    assert frame.shape[1] == 128
    assert frame.index.get_level_values("datetime").year.unique().tolist() == [2008]
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src resources/.venv/bin/python -m unittest tests.test_proxy_qlib -v
```

Expected: missing `ProxyFactorConfig` and query adapter.

- [ ] **Step 3: Implement validated configuration**

`ProxyFactorConfig` must reject years outside the local calendar, non-`all` universe values for v0, shard sizes that do not divide 128, missing provider paths, and output roots that overlap tracked source, test or configuration directories. The intended ignored output root `resources/data/proxy_factor_v0` is valid. Its fingerprint must be SHA-256 over canonical JSON containing the factor registry, label registry and all generation options.

The authoritative JSON must contain:

```json
{
  "provider_uri": "resources/data/qlib/cn_data",
  "output_root": "resources/data/proxy_factor_v0",
  "start_year": 2008,
  "end_year": 2025,
  "universe": "all",
  "frequency": "day",
  "warmup_trading_days": 120,
  "factor_shard_size": 32,
  "compression": "zstd",
  "min_global_valid_ratio": 0.05,
  "max_near_constant_ratio": 0.99
}
```

- [ ] **Step 4: Implement lazy Qlib initialization and shard queries**

```python
class QlibProxyProvider:
    def __init__(self, provider_uri: Path) -> None:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D

        qlib.init(provider_uri=str(provider_uri), region=REG_CN)
        self._features = D.features
        self._instruments = D.instruments("all")

    def query(self, fields, names, start_time, end_time):
        return self._features(
            self._instruments,
            fields=list(fields),
            start_time=start_time,
            end_time=end_time,
            freq="day",
            disk_cache=0,
        ).rename(columns=dict(zip(fields, names)))
```

`query_factor_year` must split the registry into four 32-factor calls, reject duplicate indices or columns, outer-join on the Qlib index, sort deterministically, and trim to January 1 through December 31 of the requested year. `query_label_year` must query through 20 trading days after the target year and trim label rows back to the target year.

- [ ] **Step 5: Run fake-provider tests and one real 5-day query**

```bash
PYTHONPATH=src resources/.venv/bin/python -m unittest tests.test_proxy_qlib -v
PYTHONPATH=src resources/.venv/bin/python -m factorpanel_data.proxy_cli sample-query --config configs/proxy_factor_v0.json --start 2008-01-02 --end 2008-01-08
```

Expected: 128 columns, non-empty `instrument/datetime` index, and no date after 2008-01-08.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/factorpanel_data/proxy_config.py src/factorpanel_data/qlib_proxy.py configs/proxy_factor_v0.json pyproject.toml tests/test_proxy_qlib.py
git commit -m "feat: add Qlib proxy factor adapter"
```

## Task 3: Atomic Yearly Materialization and Resume State

**Files:**
- Create: `src/factorpanel_data/proxy_materialize.py`
- Create: `tests/test_proxy_materialize.py`

- [ ] **Step 1: Write failing atomic-publication tests**

```python
def test_materialize_year_publishes_factor_and_label_partitions(tmp_path, fake_provider):
    result = materialize_year(test_config(tmp_path), fake_provider, 2008)
    assert result.factor_path == tmp_path / "factors/year=2008/part.parquet"
    assert result.label_path == tmp_path / "labels/year=2008/part.parquet"
    assert result.factor_rows > 0
    assert result.factor_columns == 128
    assert not list(tmp_path.rglob("*.tmp"))


def test_failure_does_not_replace_existing_partition(tmp_path, failing_writer):
    existing = seed_partition(tmp_path, year=2008)
    before = sha256_file(existing)
    with self.assertRaises(RuntimeError):
        materialize_year(test_config(tmp_path), fake_provider(), 2008, writer=failing_writer)
    assert sha256_file(existing) == before
    assert not list(tmp_path.rglob("*.tmp"))
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src resources/.venv/bin/python -m unittest tests.test_proxy_materialize -v
```

- [ ] **Step 3: Implement normalization and atomic publication**

```python
def normalize_proxy_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    ordered = frame.loc[:, list(columns)].replace([np.inf, -np.inf], np.nan)
    ordered = ordered.astype("float32")
    ordered.index = ordered.index.set_names(["asset", "date"])
    result = ordered.reset_index().sort_values(["date", "asset"], kind="stable")
    if result.duplicated(["date", "asset"]).any():
        raise ValueError("proxy partition contains duplicate date/asset keys")
    return result


def atomic_parquet_write(frame: pd.DataFrame, destination: Path, compression: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False, compression=compression)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
```

`materialize_year` must write factors and labels independently, compute SHA-256 after publication, and update `_state.json` atomically only after both files pass schema checks.

- [ ] **Step 4: Implement safe resume behavior**

When `--resume` is active, skip a year only if `_state.json` contains the same config fingerprint and both recorded checksums match current files. A mismatched fingerprint or checksum must raise and require `--force-year YYYY`; it must never silently combine configurations.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=src resources/.venv/bin/python -m unittest tests.test_proxy_materialize -v
git diff --check
git add src/factorpanel_data/proxy_materialize.py tests/test_proxy_materialize.py
git commit -m "feat: materialize proxy factors atomically"
```

## Task 4: Quality Statistics and Final Verification

**Files:**
- Create: `src/factorpanel_data/proxy_quality.py`
- Create: `tests/test_proxy_quality.py`

- [ ] **Step 1: Write failing quality-gate tests**

```python
def test_partition_report_detects_nonfinite_duplicate_and_constant_data():
    report = inspect_factor_partition(bad_factor_frame(), expected_factors())
    assert report.duplicate_keys == 1
    assert report.nonfinite_values == 1
    assert report.factors["pf_constant"].near_constant_ratio == 1.0


def test_finalize_rejects_missing_year_and_low_coverage(tmp_path):
    with self.assertRaisesRegex(ValueError, "missing years"):
        finalize_dataset(config_for(tmp_path), completed_years=range(2008, 2025))
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src resources/.venv/bin/python -m unittest tests.test_proxy_quality -v
```

- [ ] **Step 3: Implement partition and global reports**

For every factor calculate `valid_count`, `total_count`, `valid_ratio`, finite mean/std/min/max and `near_constant_ratio`. Near-constant means the most frequent finite float32 value occupies the stated share. Aggregate counts, first and second moments across years rather than averaging yearly ratios.

- [ ] **Step 4: Implement causality and year-boundary verification**

The causality verifier must use a synthetic provider: compute factors, modify only dates after a cutoff, recompute, and compare all values at or before the cutoff with `rtol=0`, `atol=0`, including null locations. The boundary verifier must compare December/January values from yearly generation against a single cross-year query for all 120-day factors.

- [ ] **Step 5: Implement final manifest publication**

`finalize_dataset` must require all 18 factor and label partitions, verify checksums and schemas, enforce 5% minimum global valid ratio and 99% maximum near-constant ratio, then atomically write `quality_report.json` followed by `manifest.json`. The manifest is the final completion marker and must not exist for a partial dataset.

- [ ] **Step 6: Run tests and commit**

```bash
PYTHONPATH=src resources/.venv/bin/python -m unittest tests.test_proxy_quality -v
git add src/factorpanel_data/proxy_quality.py tests/test_proxy_quality.py
git commit -m "feat: verify proxy factor quality"
```

## Task 5: Column-Projected ProxyFactor Store

**Files:**
- Create: `src/factorpanel_data/proxy_store.py`
- Create: `tests/test_proxy_store.py`
- Modify: `src/factorpanel_data/__init__.py`

- [ ] **Step 1: Write failing projected-read tests**

```python
def test_store_reads_one_factor_without_loading_other_columns(tmp_path):
    store = ProxyFactorStore(seed_complete_dataset(tmp_path))
    frame = store.read_factor(
        "pf_roc_20",
        start_date="2020-01-01",
        end_date="2020-03-31",
        assets=("SH600000", "SZ000001"),
    )
    assert list(frame.columns) == ["date", "asset", "pf_roc_20"]
    assert set(frame["asset"]) <= {"SH600000", "SZ000001"}


def test_store_builds_bounded_panel(tmp_path):
    panel = ProxyFactorStore(seed_complete_dataset(tmp_path)).read_panel(
        factor="pf_roc_20",
        end_date="2020-12-31",
        context_length=256,
        max_assets=512,
        seed=7,
    )
    assert panel.values.shape[0] == 256
    assert panel.values.shape[1] <= 512
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src /opt/miniconda3/envs/factorpanel-fm/bin/python -m unittest tests.test_proxy_store -v
```

- [ ] **Step 3: Implement PyArrow projected reads**

Use `pyarrow.dataset.dataset(root / "factors", format="parquet", partitioning="hive")`, request only `date`, `asset`, and the selected factor, and apply date filters before materialization. Validate the selected factor against `manifest.json`; never read all 128 factor columns for a single-factor request.

`read_panel` must sort available assets, sample deterministically with the provided seed only when more than `max_assets` are valid, pivot to `[T,N]`, return values plus an observed mask, and reject requests with fewer than `context_length` trading dates.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=src /opt/miniconda3/envs/factorpanel-fm/bin/python -m unittest tests.test_proxy_store -v
git add src/factorpanel_data/proxy_store.py src/factorpanel_data/__init__.py tests/test_proxy_store.py
git commit -m "feat: add projected proxy factor store"
```

## Task 6: CLI and Operational Status

**Files:**
- Create: `src/factorpanel_data/proxy_cli.py`
- Create: `tests/test_proxy_cli.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing subprocess tests**

```python
def test_status_reports_missing_years_as_json(tmp_path):
    result = run_cli("status", "--config", write_config(tmp_path), "--json")
    payload = json.loads(result.stdout)
    assert payload["complete"] is False
    assert payload["missing_factor_years"] == list(range(2008, 2026))


def test_generate_requires_explicit_resume_or_force_for_existing_output(tmp_path):
    seed_mismatched_state(tmp_path)
    result = run_cli("generate", "--config", write_config(tmp_path), check=False)
    assert result.returncode != 0
    assert "fingerprint" in result.stderr
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src resources/.venv/bin/python -m unittest tests.test_proxy_cli -v
```

- [ ] **Step 3: Implement commands**

The parser must expose:

```text
factorpanel-proxy generate --config ... --start-year ... --end-year ... [--resume] [--force-year YYYY]
factorpanel-proxy verify --config ... [--causality] [--boundary]
factorpanel-proxy status --config ... --json
factorpanel-proxy sample --config ... --factor pf_roc_20 --end-date 2025-12-31 --max-assets 512
factorpanel-proxy sample-query --config ... --start ... --end ...
```

Every command must emit one JSON object on stdout and errors on stderr with a non-zero exit code. `generate` must report the current year before each long query so progress remains observable.

- [ ] **Step 4: Run CLI tests and full suite**

```bash
PYTHONPATH=src resources/.venv/bin/python -m unittest tests.test_proxy_cli -v
PYTHONPATH=src /opt/miniconda3/envs/factorpanel-fm/bin/python -m unittest discover -s tests
/opt/miniconda3/envs/factorpanel-fm/bin/ruff check src
```

- [ ] **Step 5: Commit Task 6**

```bash
git add src/factorpanel_data/proxy_cli.py tests/test_proxy_cli.py pyproject.toml
git commit -m "feat: add ProxyFactor generation CLI"
```

## Task 7: Generate and Validate the 2008 Acceptance Partition

**Files:**
- Generate: `resources/data/proxy_factor_v0/factors/year=2008/part.parquet`
- Generate: `resources/data/proxy_factor_v0/labels/year=2008/part.parquet`
- Generate: `resources/data/proxy_factor_v0/_state.json`

- [ ] **Step 1: Run a five-day real-data query**

```bash
PYTHONPATH=src resources/.venv/bin/python -m factorpanel_data.proxy_cli sample-query \
  --config configs/proxy_factor_v0.json \
  --start 2008-01-02 \
  --end 2008-01-08
```

Expected: JSON reports 128 factors, non-zero rows, unique keys and zero infinite values.

- [ ] **Step 2: Generate the complete 2008 partition**

```bash
PYTHONPATH=src resources/.venv/bin/python -m factorpanel_data.proxy_cli generate \
  --config configs/proxy_factor_v0.json \
  --start-year 2008 \
  --end-year 2008
```

Expected: both 2008 Parquet files are atomically published and `_state.json` records their checksums.

- [ ] **Step 3: Run acceptance verification**

```bash
PYTHONPATH=src resources/.venv/bin/python -m factorpanel_data.proxy_cli verify \
  --config configs/proxy_factor_v0.json \
  --year 2008 \
  --causality \
  --boundary
```

Expected: 128/128 factors present, no duplicate keys or infinities, causality passes, and 2008 year-start values match a cross-year calculation.

- [ ] **Step 4: Inspect storage before expansion**

```bash
du -sh resources/data/proxy_factor_v0
df -h resources/data/proxy_factor_v0
```

Estimate the remaining 17-year footprint from the 2008 partition. Continue only if estimated total plus 20% headroom fits available disk.

## Task 8: Generate and Audit the Full 2008–2025 Dataset

**Files:**
- Generate: 18 factor partitions under `resources/data/proxy_factor_v0/factors/`
- Generate: 18 label partitions under `resources/data/proxy_factor_v0/labels/`
- Generate: `resources/data/proxy_factor_v0/quality_report.json`
- Generate: `resources/data/proxy_factor_v0/manifest.json`

- [ ] **Step 1: Resume full generation**

```bash
PYTHONPATH=src resources/.venv/bin/python -m factorpanel_data.proxy_cli generate \
  --config configs/proxy_factor_v0.json \
  --start-year 2008 \
  --end-year 2025 \
  --resume
```

Keep the process attached and report progress at least once per completed year. Do not declare completion while the process is still running.

- [ ] **Step 2: Run full verification and publish final manifest**

```bash
PYTHONPATH=src resources/.venv/bin/python -m factorpanel_data.proxy_cli verify \
  --config configs/proxy_factor_v0.json \
  --causality \
  --boundary \
  --finalize
```

Expected: `complete=true`, years exactly 2008–2025, 128 factors, three labels, all checksums valid, and all quality gates pass.

- [ ] **Step 3: Exercise a model-shaped projected read**

```bash
PYTHONPATH=src resources/.venv/bin/python -m factorpanel_data.proxy_cli sample \
  --config configs/proxy_factor_v0.json \
  --factor pf_roc_20 \
  --end-date 2025-12-31 \
  --context-length 256 \
  --max-assets 512 \
  --seed 7
```

Expected: output shape `[256, N]` with `1 <= N <= 512`, finite observed values, a boolean mask, sorted dates and unique assets.

- [ ] **Step 4: Run final code and artifact verification**

```bash
PYTHONPATH=src /opt/miniconda3/envs/factorpanel-fm/bin/python -m unittest discover -s tests
/opt/miniconda3/envs/factorpanel-fm/bin/ruff check src
/opt/miniconda3/envs/factorpanel-fm/bin/python -m compileall -q src
git diff --check
git status --short
```

Expected: all tests pass, lint/compile/diff checks are clean, generated data remains ignored, and only intended source/config commits are tracked.

- [ ] **Step 5: Commit any final source-only corrections**

Only if verification required source changes:

```bash
git add src tests configs pyproject.toml
git commit -m "fix: finalize ProxyFactor v0 generation"
```

Do not commit Parquet files, `_state.json`, `manifest.json`, or `quality_report.json`; they are local generated artifacts.
