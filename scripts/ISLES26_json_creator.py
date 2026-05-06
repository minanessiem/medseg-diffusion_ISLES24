import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


SESSION = "ses-1"
ANAT_DIR = "anat"
TRAINING_ROOT_CANDIDATES = (
    Path("atlas21_training_raw") / "Training_Raw",
    Path("Training_Raw"),
    Path("."),
)
DAYS_POST_STROKE_COLUMN = "DAYS_POST_STROKE"
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
TEST_SPLIT_BALANCE_COLUMNS = ("DAYS_POST_STROKE_BIN", "CHRONICITY", "ATLAS2_DATASET")


@dataclass(frozen=True)
class Isles26Case:
    case_id: str
    site_id: str
    t1_path: Path
    label_path: Path
    metadata_path: Path


def resolve_training_root(dataset_path: Path) -> Path:
    """Resolve the directory containing the site folders, e.g. R001/R034."""
    for candidate in TRAINING_ROOT_CANDIDATES:
        training_root = dataset_path / candidate
        if _contains_site_directories(training_root):
            return training_root

    raise FileNotFoundError(
        "Could not find ISLES26 training root. Expected site folders such as "
        f"'R001' under one of: {', '.join(str(dataset_path / c) for c in TRAINING_ROOT_CANDIDATES)}"
    )


def _contains_site_directories(path: Path) -> bool:
    return path.is_dir() and any(child.is_dir() and child.name.startswith("R") for child in path.iterdir())


def discover_cases(training_root: Path) -> List[Isles26Case]:
    """Discover ISLES26 cases and expected file paths without applying split logic."""
    cases = []

    for site_dir in sorted(_iter_site_dirs(training_root), key=_site_sort_key):
        for subject_dir in sorted(_iter_subject_dirs(site_dir), key=_subject_sort_key):
            anat_dir = subject_dir / SESSION / ANAT_DIR
            case_id = subject_dir.name

            cases.append(
                Isles26Case(
                    case_id=case_id,
                    site_id=site_dir.name,
                    t1_path=anat_dir / f"{case_id}_{SESSION}_space-orig_desc-brain_T1w.nii.gz",
                    label_path=anat_dir / f"{case_id}_{SESSION}_space-orig_label-lesion_desc-T1lesion_mask.nii.gz",
                    metadata_path=anat_dir / f"{case_id}_{SESSION}_metadata.csv",
                )
            )

    return cases


def _iter_site_dirs(training_root: Path) -> Iterable[Path]:
    return (child for child in training_root.iterdir() if child.is_dir() and child.name.startswith("R"))


def _iter_subject_dirs(site_dir: Path) -> Iterable[Path]:
    return (child for child in site_dir.iterdir() if child.is_dir() and child.name.startswith("sub-"))


def _site_sort_key(path: Path) -> int:
    return _extract_first_int(path.name)


def _subject_sort_key(path: Path) -> tuple[int, int]:
    match = re.match(r"sub-r(\d+)s(\d+)", path.name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return -1, _extract_first_int(path.name)


def _extract_first_int(value: str) -> int:
    digits = "".join(char for char in value if char.isdigit())
    return int(digits) if digits else -1


def validate_case(case: Isles26Case, check_valid_mask: bool = False) -> bool:
    """Return True when all required files exist and the mask is optionally non-empty."""
    required_paths = (case.t1_path, case.label_path, case.metadata_path)
    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print(f"Skipping {case.case_id}: missing {', '.join(str(path) for path in missing_paths)}")
        return False

    if check_valid_mask and not contains_valid_mask(case.label_path):
        return False

    return True


def contains_valid_mask(mask_path: Path) -> bool:
    """Check whether a NIfTI lesion mask contains at least one positive voxel."""
    lesion_mask = nib.load(str(mask_path))
    lesion_data = lesion_mask.get_fdata()

    if np.any(lesion_data > 0):
        return True

    print(f"Skipping {mask_path}: no positive lesion labels found.")
    return False


def load_metadata_row(metadata_path: Path) -> Dict[str, Any]:
    """Load the first row of an ISLES26 metadata CSV."""
    metadata_df = pd.read_csv(metadata_path)

    if DAYS_POST_STROKE_COLUMN in metadata_df.columns:
        metadata_df[DAYS_POST_STROKE_COLUMN] = pd.to_numeric(
            metadata_df[DAYS_POST_STROKE_COLUMN],
            errors="coerce",
        )

    if "CHRONICITY" in metadata_df.columns:
        metadata_df["CHRONICITY"] = pd.to_numeric(
            metadata_df["CHRONICITY"].replace("", 0).fillna(0),
            errors="coerce",
        ).fillna(0).astype(int)

    return metadata_df.iloc[0].to_dict()


def assign_round_robin_folds(cases: List[Isles26Case], num_folds: int) -> Dict[str, int]:
    """Assign folds by sorted case order until ISLES26 stratification is implemented."""
    return {case.case_id: index % num_folds for index, case in enumerate(cases)}


def build_metadata_dataframe(cases: List[Isles26Case], training_root: Path) -> pd.DataFrame:
    """Build the case-level metadata frame used for split assignment and reporting."""
    rows = []
    for case in cases:
        metadata = load_metadata_row(case.metadata_path)
        metadata["caseID"] = case.case_id
        metadata["siteID"] = case.site_id
        metadata["metadata_csv"] = case.metadata_path.relative_to(training_root).as_posix()
        rows.append(metadata)

    metadata_df = pd.DataFrame(rows)

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

    return metadata_df


def assign_site_balanced_test_split(
    metadata_df: pd.DataFrame,
    test_set_percent: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign a train/test split with site coverage in both sets."""
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


def score_test_split_balance(metadata_df: pd.DataFrame) -> float:
    """Score train/test representativeness by summed percentage-point deviations."""
    if "split" not in metadata_df.columns:
        raise ValueError("A split column is required to score train/test balance.")

    score = 0.0
    for column in _existing_columns(metadata_df, TEST_SPLIT_BALANCE_COLUMNS):
        comparison = build_split_distribution_comparison(metadata_df, column)
        score += comparison["test_minus_overall_pp"].abs().sum()

    return float(score)


def build_train_test_fold_mapping(split_df: pd.DataFrame) -> Dict[str, int]:
    """Map train cases to fold 0 and test cases to fold 1."""
    return {
        row.caseID: 1 if row.split == "test" else 0
        for row in split_df[["caseID", "split"]].itertuples(index=False)
    }


def build_case_entry(case: Isles26Case, training_root: Path, fold: int) -> Dict[str, Any]:
    """Build one MSD-style JSON entry using paths relative to the training root."""
    metadata = load_metadata_row(case.metadata_path)

    return {
        "fold": fold,
        "caseID": case.case_id,
        "siteID": case.site_id,
        "T1": [_relative_posix(case.t1_path, training_root)],
        "label": _relative_posix(case.label_path, training_root),
        "metadata_csv": _relative_posix(case.metadata_path, training_root),
        "metadata": {
            "ATLAS2_DATASET": metadata.get("ATLAS2_DATASET"),
            "SESSION_ID": metadata.get("SESSION_ID"),
            "DAYS_POST_STROKE": _json_safe_value(metadata.get("DAYS_POST_STROKE")),
            "CHRONICITY": _json_safe_value(metadata.get("CHRONICITY")),
            "SITE": metadata.get("SITE", case.site_id),
        },
    }


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


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
    print("JSON fold mapping: train -> fold 0, test -> fold 1.")

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


def create_directory_list(
    dataset_path: Path,
    num_folds: int = 5,
    check_valid_mask: bool = False,
    test_set_size: Optional[float] = None,
    seed: int = 42,
    num_split_seeds: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Create an ISLES26 dataset JSON description."""
    training_root = resolve_training_root(dataset_path)
    discovered_cases = discover_cases(training_root)
    valid_cases = [case for case in discovered_cases if validate_case(case, check_valid_mask)]

    if test_set_size is not None:
        metadata_df = build_metadata_dataframe(valid_cases, training_root)
        split_df = select_site_balanced_test_split(
            metadata_df=metadata_df,
            test_set_percent=test_set_size,
            seed=seed,
            num_split_seeds=num_split_seeds,
        )
        fold_by_case_id = build_train_test_fold_mapping(split_df)
        fold_assignment_description = "site-balanced train/test split (train=fold 0, test=fold 1)"
    else:
        split_df = None
        fold_by_case_id = assign_round_robin_folds(valid_cases, num_folds)
        fold_assignment_description = "round-robin placeholder"

    training_entries = [
        build_case_entry(case, training_root, fold_by_case_id[case.case_id])
        for case in valid_cases
    ]

    print(f"Resolved training root: {training_root}")
    print(f"Discovered cases: {len(discovered_cases)}")
    print(f"Valid cases: {len(valid_cases)}")
    print(f"Fold assignment: {fold_assignment_description}")

    if split_df is not None:
        print_test_split_report(split_df, test_set_size)

    return {"training": training_entries}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an ISLES26 JSON dataset description.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset root or Training_Raw directory")
    parser.add_argument("output_file", type=Path, help="Path to the output JSON file")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument(
        "--test_set_size",
        type=float,
        help="Optional percentage of cases to assign to fold 1 as a site-balanced test set",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test assignment")
    parser.add_argument(
        "--num_split_seeds",
        type=int,
        help="Generate and score N candidate split seeds from --seed, then keep the best split",
    )
    parser.add_argument("--check_valid_mask", action="store_true", help="Filter out cases with empty lesion masks")
    args = parser.parse_args()

    if args.num_folds < 1:
        parser.error("--num_folds must be at least 1")
    if args.test_set_size is not None and not 0 < args.test_set_size < 100:
        parser.error("--test_set_size must be between 0 and 100")
    if args.num_split_seeds is not None and args.num_split_seeds < 1:
        parser.error("--num_split_seeds must be at least 1")
    if args.num_split_seeds is not None and args.test_set_size is None:
        parser.error("--num_split_seeds requires --test_set_size")

    directory_list = create_directory_list(
        dataset_path=args.dataset_path,
        num_folds=args.num_folds,
        check_valid_mask=args.check_valid_mask,
        test_set_size=args.test_set_size,
        seed=args.seed,
        num_split_seeds=args.num_split_seeds,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as output_file:
        json.dump(directory_list, output_file, indent=4)

    print(f"Directory structure saved to {args.output_file}")


if __name__ == "__main__":
    main()
