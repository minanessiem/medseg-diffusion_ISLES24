import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scripts.ISLES26_json_creator import discover_cases, resolve_training_root, validate_case
except ModuleNotFoundError:
    from ISLES26_json_creator import discover_cases, resolve_training_root, validate_case


DAYS_POST_STROKE_COLUMN = "DAYS_POST_STROKE"
CATEGORICAL_COLUMNS = ("ATLAS2_DATASET", "CHRONICITY", "SITE")
TEST_SPLIT_BALANCE_COLUMNS = ("DAYS_POST_STROKE_BIN", "CHRONICITY", "ATLAS2_DATASET")
DAYS_POST_STROKE_BINS = (
    -np.inf,
    7,
    30,
    90,
    180,
    365,
    730,
    np.inf,
)
DAYS_POST_STROKE_LABELS = (
    "0-7 days",
    "8-30 days",
    "31-90 days",
    "91-180 days",
    "181-365 days",
    "366-730 days",
    ">730 days",
)


def load_metadata_data(dataset_path: Path, check_valid_mask: bool = False) -> pd.DataFrame:
    """Load all ISLES26 metadata CSVs into one case-level dataframe."""
    training_root = resolve_training_root(dataset_path)
    discovered_cases = discover_cases(training_root)
    valid_cases = [case for case in discovered_cases if validate_case(case, check_valid_mask)]

    rows = []
    for case in valid_cases:
        try:
            metadata = pd.read_csv(case.metadata_path).iloc[0].to_dict()
        except Exception as exc:
            print(f"Skipping {case.case_id}: could not read metadata CSV ({exc})")
            continue

        metadata["caseID"] = case.case_id
        metadata["siteID"] = case.site_id
        metadata["metadata_csv"] = case.metadata_path.relative_to(training_root).as_posix()
        rows.append(metadata)

    if not rows:
        raise ValueError("No ISLES26 metadata CSV files were successfully loaded.")

    metadata_df = pd.DataFrame(rows)
    metadata_df = normalize_metadata(metadata_df)

    print(f"Resolved training root: {training_root}")
    print(f"Discovered cases: {len(discovered_cases)}")
    print(f"Cases with complete required files: {len(valid_cases)}")
    print(f"Metadata rows loaded: {len(metadata_df)}")

    return metadata_df


def normalize_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and add split-planning helper columns."""
    metadata_df = metadata_df.copy()

    if DAYS_POST_STROKE_COLUMN in metadata_df.columns:
        metadata_df[DAYS_POST_STROKE_COLUMN] = pd.to_numeric(
            metadata_df[DAYS_POST_STROKE_COLUMN],
            errors="coerce",
        )
        metadata_df["DAYS_POST_STROKE_BIN"] = pd.cut(
            metadata_df[DAYS_POST_STROKE_COLUMN],
            bins=DAYS_POST_STROKE_BINS,
            labels=DAYS_POST_STROKE_LABELS,
        )

    if "CHRONICITY" in metadata_df.columns:
        metadata_df["CHRONICITY"] = pd.to_numeric(
            metadata_df["CHRONICITY"].replace("", 0).fillna(0),
            errors="coerce",
        ).fillna(0).astype(int)

    for column in CATEGORICAL_COLUMNS:
        if column in metadata_df.columns:
            metadata_df[column] = metadata_df[column].replace("", pd.NA)

    return metadata_df


def print_metadata_report(
    metadata_df: pd.DataFrame,
    num_folds: Optional[int] = None,
    test_set_size: Optional[float] = None,
    seed: int = 42,
    num_split_seeds: Optional[int] = None,
) -> pd.DataFrame:
    """Print descriptive metadata statistics, with optional fold-oriented hints."""
    print_header("Dataset Overview")
    print(f"Total cases: {len(metadata_df)}")
    print(f"Columns: {', '.join(metadata_df.columns)}")
    print(f"Unique case IDs: {metadata_df['caseID'].nunique()}")
    print(f"Unique session IDs: {_safe_nunique(metadata_df, 'SESSION_ID')}")
    print(f"Unique sites: {_safe_nunique(metadata_df, 'SITE')}")

    print_header("Missing Values")
    missing_summary = build_missing_summary(metadata_df)
    if missing_summary.empty:
        print("No missing values detected.")
    else:
        print(missing_summary.to_string())

    print_header("Categorical Distributions")
    for column in _existing_columns(metadata_df, ["ATLAS2_DATASET", "SITE", "siteID", "CHRONICITY"]):
        print_value_counts(metadata_df, column)

    print_header("Days Post Stroke")
    if DAYS_POST_STROKE_COLUMN in metadata_df.columns:
        print_numeric_summary(metadata_df[DAYS_POST_STROKE_COLUMN], DAYS_POST_STROKE_COLUMN)
        print_value_counts(metadata_df, "DAYS_POST_STROKE_BIN")
    else:
        print(f"{DAYS_POST_STROKE_COLUMN} column not found.")

    print_header("Cross Tabulations")
    print_cross_tab(metadata_df, "SITE", "DAYS_POST_STROKE_BIN")
    print_cross_tab(metadata_df, "siteID", "DAYS_POST_STROKE_BIN")
    print_cross_tab(metadata_df, "SITE", "CHRONICITY")

    if num_folds is not None:
        print_header("Fold Planning Hints")
        print_fold_planning_hints(metadata_df, num_folds)

    if test_set_size is not None:
        metadata_df = select_site_balanced_test_split(
            metadata_df=metadata_df,
            test_set_percent=test_set_size,
            seed=seed,
            num_split_seeds=num_split_seeds,
        )
        print_test_split_report(metadata_df, test_set_size)

    return metadata_df


def build_missing_summary(metadata_df: pd.DataFrame) -> pd.DataFrame:
    missing_count = metadata_df.isna().sum()
    missing_percent = (missing_count / len(metadata_df)) * 100
    summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percent": missing_percent.round(2),
        }
    )
    return summary[summary["missing_count"] > 0].sort_values(
        by=["missing_percent", "missing_count"],
        ascending=False,
    )


def print_numeric_summary(series: pd.Series, label: str) -> None:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()

    print(f"\n{label}:")
    print(f"  valid_count: {len(numeric_series)}")
    print(f"  missing_count: {series.isna().sum()}")

    if numeric_series.empty:
        print("  No valid numeric values.")
        return

    summary = numeric_series.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    for stat_name, value in summary.items():
        print(f"  {stat_name:>8}: {value:.2f}")


def print_value_counts(metadata_df: pd.DataFrame, column: str) -> None:
    if column not in metadata_df.columns:
        return

    display_values = _display_series(metadata_df[column])
    counts = display_values.value_counts(dropna=False)
    percentages = display_values.value_counts(dropna=False, normalize=True) * 100
    summary = pd.DataFrame(
        {
            "count": counts,
            "percent": percentages.round(2),
        }
    )

    print(f"\n{column}:")
    print(summary.to_string())


def print_cross_tab(metadata_df: pd.DataFrame, row_column: str, column_column: str) -> None:
    if row_column not in metadata_df.columns or column_column not in metadata_df.columns:
        print(f"\nSkipping {row_column} x {column_column}: required columns not found.")
        return

    print(f"\n{row_column} x {column_column}:")
    crosstab = pd.crosstab(
        _display_series(metadata_df[row_column]),
        _display_series(metadata_df[column_column]),
        dropna=False,
    )
    print(crosstab.to_string())


def print_fold_planning_hints(metadata_df: pd.DataFrame, num_folds: int) -> None:
    print(f"Requested folds: {num_folds}")
    print(f"Ideal cases per fold: {len(metadata_df) / num_folds:.2f}")

    for column in _existing_columns(metadata_df, ["SITE", "siteID", "DAYS_POST_STROKE_BIN", "CHRONICITY"]):
        counts = _display_series(metadata_df[column]).value_counts(dropna=False)
        scarce_groups = counts[counts < num_folds]

        print(f"\n{column}:")
        print(f"  groups: {len(counts)}")
        print(f"  largest_group: {counts.index[0]} ({counts.iloc[0]} cases)")
        if scarce_groups.empty:
            print("  every group has at least one case per fold in principle")
        else:
            print("  groups with fewer cases than requested folds:")
            for value, count in scarce_groups.items():
                print(f"    {value}: {count}")

    print("\nInterpretation:")
    print("- `SITE`/`siteID` are likely important because scanner/site effects can leak into model behavior.")
    print("- `DAYS_POST_STROKE_BIN` is the clearest continuous clinical axis available in this metadata.")
    print("- `CHRONICITY` is normalized as 0/1, with missing or blank values treated as 0.")
    print("- Small site groups may need grouped, site-aware, or repeated split strategies instead of strict stratification.")


def assign_site_balanced_test_split(
    metadata_df: pd.DataFrame,
    test_set_percent: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign a simulated test split with site coverage in train and test."""
    if "SITE" not in metadata_df.columns:
        raise ValueError("SITE column is required for site-balanced train/test assignment.")

    split_df = metadata_df.copy()
    total_cases = len(split_df)
    target_test_count = int(round(total_cases * (test_set_percent / 100.0)))
    site_counts = split_df["SITE"].value_counts().sort_index()

    min_test_count = len(site_counts)
    max_test_count = total_cases - len(site_counts)
    if target_test_count < min_test_count:
        raise ValueError(
            f"{test_set_percent}% gives {target_test_count} test cases, but at least "
            f"{min_test_count} are needed to place every site in the test set."
        )
    if target_test_count > max_test_count:
        raise ValueError(
            f"{test_set_percent}% gives {target_test_count} test cases, but at most "
            f"{max_test_count} are allowed while keeping every site in training."
        )
    if (site_counts < 2).any():
        sparse_sites = site_counts[site_counts < 2].index.tolist()
        raise ValueError(f"Cannot place these sites in both train and test: {sparse_sites}")

    site_quotas = allocate_group_quotas(
        counts=site_counts,
        target=target_test_count,
        min_per_group=1,
        max_per_group=site_counts - 1,
    )

    rng = np.random.default_rng(seed)
    test_indices = []
    for site, quota in site_quotas.items():
        site_df = split_df[split_df["SITE"] == site]
        test_indices.extend(select_representative_site_cases(site_df, quota, rng))

    split_df["split"] = "train"
    split_df.loc[test_indices, "split"] = "test"
    split_df.attrs["split_seed"] = seed
    split_df.attrs["split_balance_score"] = score_test_split_balance(split_df)
    split_df.attrs["num_split_candidates"] = 1
    return split_df


def select_site_balanced_test_split(
    metadata_df: pd.DataFrame,
    test_set_percent: float,
    seed: int = 42,
    num_split_seeds: Optional[int] = None,
) -> pd.DataFrame:
    """Choose one deterministic split, optionally by searching generated candidate seeds."""
    if num_split_seeds is None:
        return assign_site_balanced_test_split(metadata_df, test_set_percent, seed=seed)

    seed_rng = np.random.default_rng(seed)
    candidate_seeds = seed_rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=num_split_seeds,
        dtype=np.uint32,
    )

    best_split = None
    best_score = np.inf
    for candidate_seed in tqdm(candidate_seeds, desc="Evaluating split seeds", unit="seed"):
        candidate_split = assign_site_balanced_test_split(
            metadata_df,
            test_set_percent,
            seed=int(candidate_seed),
        )
        candidate_score = candidate_split.attrs["split_balance_score"]
        if candidate_score < best_score:
            best_split = candidate_split
            best_score = candidate_score

    if best_split is None:
        raise ValueError("No split candidates were generated.")

    best_split.attrs["base_seed"] = seed
    best_split.attrs["num_split_candidates"] = num_split_seeds
    return best_split


def score_test_split_balance(metadata_df: pd.DataFrame) -> float:
    """Score train/test representativeness by summed percentage-point deviations."""
    if "split" not in metadata_df.columns:
        raise ValueError("A split column is required to score train/test balance.")

    score = 0.0
    for column in _existing_columns(metadata_df, TEST_SPLIT_BALANCE_COLUMNS):
        comparison = build_split_distribution_comparison(metadata_df, column)
        score += comparison["test_minus_overall_pp"].abs().sum()

    return float(score)


def allocate_group_quotas(
    counts: pd.Series,
    target: int,
    min_per_group: int,
    max_per_group: pd.Series,
) -> pd.Series:
    """Allocate integer quotas proportionally while respecting group bounds."""
    proportions = counts / counts.sum()
    ideal = proportions * target
    quotas = np.floor(ideal).astype(int)
    quotas = quotas.clip(lower=min_per_group)
    quotas = pd.Series(np.minimum(quotas, max_per_group), index=counts.index)

    while quotas.sum() < target:
        eligible = quotas[quotas < max_per_group]
        if eligible.empty:
            break
        remainders = (ideal - quotas).loc[eligible.index].sort_values(ascending=False)
        quotas.loc[remainders.index[0]] += 1

    while quotas.sum() > target:
        eligible = quotas[quotas > min_per_group]
        if eligible.empty:
            break
        remainders = (ideal - quotas).loc[eligible.index].sort_values(ascending=True)
        quotas.loc[remainders.index[0]] -= 1

    return quotas.astype(int)


def select_representative_site_cases(
    site_df: pd.DataFrame,
    quota: int,
    rng: np.random.Generator,
) -> List[int]:
    """Sample cases within a site across available time/chronicity strata."""
    stratification_columns = _existing_columns(site_df, ["DAYS_POST_STROKE_BIN", "CHRONICITY"])
    if not stratification_columns:
        return rng.choice(site_df.index.to_numpy(), size=quota, replace=False).tolist()

    stratum_labels = site_df[stratification_columns].apply(
        lambda row: " | ".join(str(value) if pd.notna(value) else "Missing" for value in row),
        axis=1,
    )
    stratum_counts = stratum_labels.value_counts()
    stratum_quotas = allocate_group_quotas(
        counts=stratum_counts,
        target=quota,
        min_per_group=0,
        max_per_group=stratum_counts,
    )

    selected_indices = []
    for stratum, stratum_quota in stratum_quotas.items():
        if stratum_quota == 0:
            continue
        candidate_indices = site_df.index[stratum_labels == stratum].to_numpy()
        selected_indices.extend(
            rng.choice(candidate_indices, size=stratum_quota, replace=False).tolist()
        )

    return selected_indices


def print_test_split_report(metadata_df: pd.DataFrame, test_set_size: float) -> None:
    print_header("Train/Test Split Simulation")
    split_counts = metadata_df["split"].value_counts()
    test_count = int(split_counts.get("test", 0))
    train_count = int(split_counts.get("train", 0))
    split_seed = metadata_df.attrs.get("split_seed")
    base_seed = metadata_df.attrs.get("base_seed")
    num_candidates = metadata_df.attrs.get("num_split_candidates", 1)
    balance_score = metadata_df.attrs.get("split_balance_score")

    print(f"Requested test set size: {test_set_size:.2f}%")
    print(f"Train cases: {train_count}")
    print(f"Test cases: {test_count}")
    print(f"Actual test set size: {(test_count / len(metadata_df)) * 100:.2f}%")
    if num_candidates > 1:
        print(f"Base seed: {base_seed}")
        print(f"Candidate seeds tried: {num_candidates}")
        print(f"Selected split seed: {split_seed}")
    else:
        print(f"Split seed: {split_seed}")
    if balance_score is not None:
        print(f"Balance score: {balance_score:.4f} lower is better")
    print("Assignment policy: site-proportional quotas with at least one train and one test case per site.")

    print("\nSITE coverage:")
    site_split_counts = pd.crosstab(metadata_df["SITE"], metadata_df["split"])
    site_split_counts["total"] = site_split_counts.sum(axis=1)
    site_split_counts["test_percent"] = (
        site_split_counts.get("test", 0) / site_split_counts["total"] * 100
    ).round(2)
    print(site_split_counts.to_string())

    print("\nRepresentativeness checks:")
    for column in _existing_columns(
        metadata_df,
        ["SITE", "DAYS_POST_STROKE_BIN", "CHRONICITY", "ATLAS2_DATASET"],
    ):
        print_split_distribution_comparison(metadata_df, column)

    if DAYS_POST_STROKE_COLUMN in metadata_df.columns:
        print("\nDAYS_POST_STROKE by split:")
        print(metadata_df.groupby("split")[DAYS_POST_STROKE_COLUMN].describe().round(2).to_string())


def print_split_distribution_comparison(metadata_df: pd.DataFrame, column: str) -> None:
    comparison = build_split_distribution_comparison(metadata_df, column)

    print(f"\n{column}:")
    print(comparison.to_string(index=False))


def build_split_distribution_comparison(metadata_df: pd.DataFrame, column: str) -> pd.DataFrame:
    display_values = _display_series(metadata_df[column])
    total_count = len(metadata_df)
    rows = []

    for value in display_values.value_counts().index:
        value_mask = display_values == value
        train_mask = metadata_df["split"] == "train"
        test_mask = metadata_df["split"] == "test"

        overall_count = int(value_mask.sum())
        train_count = int((value_mask & train_mask).sum())
        test_count = int((value_mask & test_mask).sum())
        overall_percent = (overall_count / total_count) * 100
        train_percent = (train_count / train_mask.sum()) * 100
        test_percent = (test_count / test_mask.sum()) * 100

        rows.append(
            {
                column: value,
                "overall_count": overall_count,
                "overall_percent": round(overall_percent, 2),
                "train_count": train_count,
                "train_percent": round(train_percent, 2),
                "test_count": test_count,
                "test_percent": round(test_percent, 2),
                "test_minus_overall_pp": round(test_percent - overall_percent, 2),
            }
        )

    return pd.DataFrame(rows)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _existing_columns(metadata_df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [column for column in columns if column in metadata_df.columns]


def _display_series(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), "Missing")


def _safe_nunique(metadata_df: pd.DataFrame, column: str) -> str:
    if column not in metadata_df.columns:
        return "column not found"
    return str(metadata_df[column].nunique(dropna=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore combined ISLES26 metadata with optional fold-planning analysis."
    )
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset root or Training_Raw directory")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--fold", type=int, help="Optional number of folds to consider for split planning")
    split_group.add_argument(
        "--test_set_size",
        type=float,
        help="Optional percentage of cases to assign to a simulated site-balanced test set",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulated train/test assignment")
    parser.add_argument(
        "--num_split_seeds",
        type=int,
        help="Generate and score N candidate split seeds from --seed, then keep the best split",
    )
    parser.add_argument("--check_valid_mask", action="store_true", help="Exclude cases with empty lesion masks")
    parser.add_argument("--output", type=Path, help="Optional path to save the combined metadata CSV")
    args = parser.parse_args()

    if args.fold is not None and args.fold < 1:
        parser.error("--fold must be at least 1")
    if args.test_set_size is not None and not 0 < args.test_set_size < 100:
        parser.error("--test_set_size must be between 0 and 100")
    if args.num_split_seeds is not None and args.num_split_seeds < 1:
        parser.error("--num_split_seeds must be at least 1")
    if args.num_split_seeds is not None and args.test_set_size is None:
        parser.error("--num_split_seeds requires --test_set_size")

    metadata_df = load_metadata_data(
        dataset_path=args.dataset_path,
        check_valid_mask=args.check_valid_mask,
    )
    metadata_df = print_metadata_report(
        metadata_df,
        num_folds=args.fold,
        test_set_size=args.test_set_size,
        seed=args.seed,
        num_split_seeds=args.num_split_seeds,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        metadata_df.to_csv(args.output, index=False)
        print(f"\nCombined metadata saved to {args.output}")


if __name__ == "__main__":
    main()
