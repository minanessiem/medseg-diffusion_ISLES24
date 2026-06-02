import os
import re
import json
import nibabel as nib
import numpy as np
import argparse
import pandas as pd
from scipy.stats import f_oneway, chi2_contingency

def get_severity_label(nihss):
    if pd.isna(nihss):
        return None
    if 0 <= nihss <= 4:
        return 'Minor/No stroke'
    elif 5 <= nihss <= 15:
        return 'Moderate'
    elif 16 <= nihss <= 20:
        return 'Moderate/Severe'
    else:
        return 'Severe'

def load_demographics_row(demo_path):
    demo_df = pd.read_csv(demo_path)

    # Convert specific columns to numeric, handling any errors
    numeric_columns = [
        'Age', 'Glucose', 'Leucocytes', 'CRP', 'INR',
        'Onset to door (minutes)', 'Door to imaging (minutes)', 'NIHSS at admission'
    ]
    for col in numeric_columns:
        if col in demo_df.columns:
            demo_df[col] = pd.to_numeric(demo_df[col], errors='coerce')

    # Convert binary/categorical variables to proper format
    binary_columns = ['Sex', 'Atrial fibrillation', 'Hypertension', 'Diabetes', 'Hyperlipidemia']
    for col in binary_columns:
        if col in demo_df.columns:
            # Convert to string and clean up
            demo_df[col] = demo_df[col].astype(str).str.lower()
            # Map variations to standardized values
            demo_df[col] = demo_df[col].map({
                'yes': 'Yes', 'y': 'Yes', 'true': 'Yes', '1': 'Yes', '1.0': 'Yes',
                'no': 'No', 'n': 'No', 'false': 'No', '0': 'No', '0.0': 'No'
            })

    return demo_df.iloc[0].to_dict()

def analyze_fold_statistics(fold_assignments, case_to_demo_path):
    fold_ids = sorted(fold_assignments.keys())

    # Collect all demographic data
    all_demo_data = {}
    for case in sorted(case_to_demo_path.keys()):
        demo_path = case_to_demo_path[case]
        try:
            all_demo_data[case] = load_demographics_row(demo_path)
        except Exception as e:
            print(f"Warning: Could not process demographics for case {case}: {str(e)}")
            continue

    demo_df = pd.DataFrame.from_dict(all_demo_data, orient='index')
    if demo_df.empty:
        print("No valid demographic data found. Skipping statistical analysis.")
        return

    # NIHSS-based fold distribution check
    if 'NIHSS at admission' in demo_df.columns:
        valid_nihss_series = pd.to_numeric(demo_df['NIHSS at admission'], errors='coerce')
        valid_nihss_cases = valid_nihss_series.dropna().index
        dropped_cases = valid_nihss_series[valid_nihss_series.isna()].index.tolist()

        print(f"\nCases with valid NIHSS: {len(valid_nihss_cases)}")
        print(f"Cases dropped due to missing NIHSS: {len(dropped_cases)}")

        print("\nNIHSS Stratification Statistics:")
        print("-" * 40)
        for fold in fold_ids:
            fold_case_ids = [case for case in fold_assignments[fold] if case in valid_nihss_cases]
            fold_nihss = valid_nihss_series.loc[fold_case_ids].dropna()
            severity_distribution = fold_nihss.apply(get_severity_label).value_counts().to_dict()
            print(f"\nFold {fold}:")
            print("Severity distribution:")
            for severity, count in severity_distribution.items():
                print(f"{severity:20}: {count:3d} cases")
            print(f"Total cases: {len(fold_nihss)}")
    else:
        print("\nNIHSS column not found in demographics. Skipping NIHSS stratification statistics.")

    # Perform statistical tests on other variables
    print("\nStatistical Analysis of Fold Distributions:")
    print("-" * 50)

    # Variables to test
    numeric_vars = ['Age', 'Glucose', 'Leucocytes', 'CRP', 'INR']
    categorical_vars = ['Sex', 'Atrial fibrillation', 'Hypertension', 'Diabetes', 'Hyperlipidemia']
    time_vars = ['Onset to door (minutes)', 'Door to imaging (minutes)']

    print("\nNumeric and Time Variables:")
    print("-" * 30)
    for var in numeric_vars + time_vars:
        if var in demo_df.columns:
            print(f"\n{var}:")
            try:
                # Prepare data for ANOVA
                fold_data = [
                    pd.to_numeric(
                        demo_df[demo_df.index.isin(fold_assignments[fold])][var],
                        errors='coerce'
                    ).dropna()
                    for fold in fold_ids
                ]

                # Print basic statistics for each fold
                for fold_idx, data in zip(fold_ids, fold_data):
                    print(f"Fold {fold_idx}: mean={data.mean():.2f}, std={data.std():.2f}, n={len(data)}")

                # Perform one-way ANOVA if we have enough data
                if all(len(d) > 0 for d in fold_data):
                    f_stat, p_value = f_oneway(*fold_data)
                    print(f"One-way ANOVA: F={f_stat:.2f}, p={p_value:.4f}")
                else:
                    print("Insufficient data for statistical test")
            except Exception as e:
                print(f"Could not perform statistical test: {str(e)}")

    print("\nCategorical Variables:")
    print("-" * 30)
    for var in categorical_vars:
        if var in demo_df.columns:
            print(f"\n{var}:")
            try:
                # Create contingency table
                contingency = pd.DataFrame([
                    demo_df[demo_df.index.isin(fold_assignments[fold])][var].value_counts()
                    for fold in fold_ids
                ])

                # Print distribution for each fold
                for fold in fold_ids:
                    fold_dist = demo_df[demo_df.index.isin(fold_assignments[fold])][var].value_counts()
                    print(f"\nFold {fold} distribution:")
                    for val, count in fold_dist.items():
                        print(f"{val}: {count}")

                # Perform chi-square test
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                print(f"\nChi-square test: χ²={chi2:.2f}, p={p_value:.4f}")
            except Exception as e:
                print(f"Could not perform statistical test: {str(e)}")

def create_stratified_folds(df, num_folds=5, nihss_column='NIHSS at admission', id_column=None):
    """
    Create stratified folds based on NIHSS severity categories.
    
    Args:
        df (pd.DataFrame): DataFrame containing the NIHSS scores
        num_folds (int): Number of folds to create. Default 5
        nihss_column (str): Name of the column containing NIHSS scores. Default 'NIHSS at admission'
        id_column (str): Optional name of ID column. If None, uses DataFrame index
    
    Returns:
        dict: Dictionary mapping fold numbers (0 to num_folds-1) to lists of case IDs
        dict: Dictionary containing fold statistics and distributions
    """
    # Create a DataFrame with case IDs and their severity labels
    stratification_df = pd.DataFrame({
        'id': df.index if id_column is None else df[id_column],
        'NIHSS': df[nihss_column],
        'severity': df[nihss_column].apply(get_severity_label)
    }).dropna()
    
    # Initialize fold assignments and statistics
    fold_assignments = {i: [] for i in range(num_folds)}
    fold_stats = {
        'total_cases': len(stratification_df),
        'cases_per_fold': len(stratification_df) / num_folds,
        'category_distributions': {},
        'fold_distributions': {}
    }
    
    # Assign cases to folds, starting with the rarest category
    for severity in ['Severe', 'Moderate/Severe', 'Minor/No stroke', 'Moderate']:
        severity_cases = stratification_df[stratification_df['severity'] == severity]
        cases_in_category = len(severity_cases)
        
        # Store category distribution
        fold_stats['category_distributions'][severity] = {
            'total_cases': cases_in_category,
            'ideal_per_fold': cases_in_category / num_folds
        }
        
        # Distribute cases across folds
        for idx, case in enumerate(severity_cases.itertuples()):
            fold = idx % num_folds
            fold_assignments[fold].append(case.id)
    
    # Calculate final distributions for each fold
    for fold in range(num_folds):
        fold_cases = stratification_df[stratification_df['id'].isin(fold_assignments[fold])]
        severity_counts = fold_cases['severity'].value_counts().to_dict()
        fold_stats['fold_distributions'][fold] = {
            'total_cases': len(fold_cases),
            'severity_distribution': severity_counts
        }
    
    return fold_assignments, fold_stats

def contains_valid_mask(mask_path):
    """
    Checks if the given NIfTI file contains at least one non-zero element (valid mask).
    
    Args:
    mask_path (str): The path to the NIfTI file.

    Returns:
    bool: True if the mask contains at least one non-zero element, False otherwise.
    """
    if not os.path.exists(mask_path):
        print(f"Mask file does not exist: {mask_path}")
        return False

    lesion_mask = nib.load(mask_path)
    lesion_data = lesion_mask.get_fdata()

    if np.any(lesion_data > 0):
        return True
    else:
        print(f"No mask labels found in {mask_path}. Skipping this case.")
        return False

def create_directory_list(base_path, num_folds=5, check_valid_mask=False, stratify=False):
    training_list = []
    fold_counter = 0
    case_to_fold = {}
    
    case_folders = sorted(
        [folder for folder in os.listdir(os.path.join(base_path, "raw_data")) if os.path.isdir(os.path.join(base_path, "raw_data", folder))],
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    
    # First pass: collect all valid cases and NIHSS scores if stratifying
    valid_cases = []
    nihss_data = {}
    
    for case_folder in case_folders:
        case_path = os.path.join(base_path, "raw_data", case_folder)
        ses_01_path = os.path.join(case_path, "ses-01")
        ses_02_path = os.path.join(case_path, "ses-02")
        
        lesion_mask_path = os.path.join(base_path, "derivatives", case_folder, "ses-02", f"{case_folder}_ses-02_lesion-msk.nii.gz")
        
        if os.path.exists(os.path.join(ses_01_path, f"{case_folder}_ses-01_ncct.nii.gz")):
            if not check_valid_mask or contains_valid_mask(lesion_mask_path):
                valid_cases.append(case_folder)
                
                if stratify:
                    # Read NIHSS score from demographics CSV
                    demographics_path = os.path.join(base_path, "phenotype", case_folder, "ses-01", f"{case_folder}_ses-01_demographic_baseline.csv")
                    try:
                        demo_df = pd.read_csv(demographics_path)
                        nihss = demo_df['NIHSS at admission'].iloc[0] if 'NIHSS at admission' in demo_df.columns else None
                        nihss_data[case_folder] = nihss
                    except:
                        nihss_data[case_folder] = None
    
    # Create fold assignments if stratifying
    if stratify:
        nihss_df = pd.DataFrame.from_dict(nihss_data, orient='index', columns=['NIHSS at admission'])
        valid_nihss_df = nihss_df.dropna()
        dropped_cases = nihss_df[nihss_df['NIHSS at admission'].isna()].index.tolist()
        
        print(f"\nCases with valid NIHSS: {len(valid_nihss_df)}")
        print(f"Cases dropped due to missing NIHSS: {len(dropped_cases)}")
        
        fold_assignments, _ = create_stratified_folds(valid_nihss_df, num_folds=num_folds)
        
        # Create case mapping using fold assignments
        for fold, cases in fold_assignments.items():
            for case_id in cases:
                case_to_fold[case_id] = fold
        
        # Filter valid_cases to only include those with NIHSS scores
        valid_cases = [case for case in valid_cases if case in case_to_fold]
        
        case_to_demo_path = {
            case: os.path.join(base_path, "phenotype", case, "ses-01", f"{case}_ses-01_demographic_baseline.csv")
            for case in valid_cases
        }
        analyze_fold_statistics(fold_assignments, case_to_demo_path)
    
    # Create the directory list
    for case_folder in valid_cases:
        case_dict = {
            "fold": case_to_fold[case_folder] if stratify else fold_counter % num_folds,
            "caseID": f"{case_folder}",
            "NCCT": [os.path.join("raw_data", case_folder, "ses-01", f"{case_folder}_ses-01_ncct.nii.gz")],
            "label": os.path.join("derivatives", case_folder, "ses-02", f"{case_folder}_ses-02_lesion-msk.nii.gz"),
            "CTA": os.path.join("derivatives", case_folder, "ses-01", f"{case_folder}_ses-01_space-ncct_cta.nii.gz"),
            "CBF": os.path.join("derivatives", case_folder, "ses-01", "perfusion-maps", f"{case_folder}_ses-01_space-ncct_cbf.nii.gz"),
            "CBV": os.path.join("derivatives", case_folder, "ses-01", "perfusion-maps", f"{case_folder}_ses-01_space-ncct_cbv.nii.gz"),
            "MTT": os.path.join("derivatives", case_folder, "ses-01", "perfusion-maps", f"{case_folder}_ses-01_space-ncct_mtt.nii.gz"),
            "TMAX": os.path.join("derivatives", case_folder, "ses-01", "perfusion-maps", f"{case_folder}_ses-01_space-ncct_tmax.nii.gz"),
            "demographics_csv": os.path.join("phenotype", case_folder, "ses-01", f"{case_folder}_ses-01_demographic_baseline.csv"),
            "outcomes_csv": os.path.join("phenotype", case_folder, "ses-02", f"{case_folder}_ses-01_outcome.csv"),
        }
        
        training_list.append(case_dict)
        
        if not stratify:
            fold_counter += 1
    
    result = {
        "training": training_list
    }
    
    return result

def check_existing_split_json(base_path, split_json_path):
    with open(split_json_path, 'r') as f:
        split_data = json.load(f)

    if 'training' not in split_data or not isinstance(split_data['training'], list):
        raise ValueError("Invalid split JSON format: expected top-level key 'training' with a list of cases.")

    fold_assignments = {}
    case_to_demo_path = {}

    for case in split_data['training']:
        case_id = case.get('caseID')
        fold = case.get('fold')

        if case_id is None or fold is None:
            print(f"Skipping malformed case entry: {case}")
            continue

        fold = int(fold)
        if fold not in fold_assignments:
            fold_assignments[fold] = []
        fold_assignments[fold].append(case_id)

        demo_rel_path = case.get('demographics_csv')
        if demo_rel_path:
            demo_path = demo_rel_path if os.path.isabs(demo_rel_path) else os.path.join(base_path, demo_rel_path)
        else:
            demo_path = os.path.join(base_path, "phenotype", case_id, "ses-01", f"{case_id}_ses-01_demographic_baseline.csv")
        case_to_demo_path[case_id] = demo_path

    if not fold_assignments:
        raise ValueError("No valid fold assignments found in the input JSON.")

    print(f"Loaded split JSON with {len(split_data['training'])} entries.")
    print(f"Detected folds: {sorted(fold_assignments.keys())}")
    analyze_fold_statistics(fold_assignments, case_to_demo_path)

def main():
    parser = argparse.ArgumentParser(description="Create a JSON file listing the dataset structure.")
    parser.add_argument("dataset_path", nargs="?", help="Path to the dataset root")
    parser.add_argument("output_file", nargs="?", help="Path to the output JSON file (generation mode)")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for cross-validation (default: 5)")
    parser.add_argument("--check_valid_mask", action="store_true", help="Check for valid masks and filter out empty ones")
    parser.add_argument("--stratify", action="store_true", help="Stratify folds based on NIHSS scores")
    parser.add_argument("--check_split_json", type=str, help="Path to an existing split JSON file to analyze")
    args = parser.parse_args()

    if args.check_split_json:
        if not args.dataset_path:
            parser.error("dataset_path is required when using --check_split_json")
        check_existing_split_json(args.dataset_path, args.check_split_json)
        return

    if not args.dataset_path or not args.output_file:
        parser.error("dataset_path and output_file are required in generation mode")

    # Create the directory list with the given parameters
    directory_list = create_directory_list(
        args.dataset_path,
        num_folds=args.num_folds,
        check_valid_mask=args.check_valid_mask,
        stratify=args.stratify
    )

    # Write the result to a JSON file
    with open(args.output_file, 'w') as f:
        json.dump(directory_list, f, indent=4)

    print(f"Directory structure saved to {args.output_file}")

if __name__ == "__main__":
    main()