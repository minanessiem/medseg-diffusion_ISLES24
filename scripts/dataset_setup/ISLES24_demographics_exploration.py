import os
import pandas as pd
import argparse
from pathlib import Path

def load_demographics_data(dataset_path):
    """
    Load all demographic CSV files from the dataset into a single DataFrame.
    
    Args:
        dataset_path (str): Path to the dataset root directory
    
    Returns:
        pd.DataFrame: Combined DataFrame with all demographic data
    """
    # Initialize empty list to store individual DataFrames
    dfs = []
    
    # Get all case folders
    phenotype_path = Path(dataset_path) / "phenotype"
    if not phenotype_path.exists():
        raise ValueError(f"Phenotype directory not found at {phenotype_path}")
    
    # Walk through all case folders
    for case_folder in sorted(os.listdir(phenotype_path)):
        demo_file = phenotype_path / case_folder / "ses-01" / f"{case_folder}_ses-01_demographic_baseline.csv"
        
        if demo_file.exists():
            # Read the CSV file
            try:
                df = pd.read_csv(demo_file)
                # Add the case ID as the first column
                df.insert(0, 'id', case_folder)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {demo_file}: {e}")
        else:
            print(f"Demographics file not found for case {case_folder}")
    
    # Combine all DataFrames
    if not dfs:
        raise ValueError("No demographic data files were successfully loaded")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def convert_time_str_to_minutes(time_str):
    """
    Convert time string in format HH:MM:SS to minutes as float.
    Invalid or missing values return 0.
    
    Args:
        time_str: String in format HH:MM:SS
    Returns:
        float: Total minutes
    """
    if pd.isna(time_str):
        return 0.0
    
    try:
        # Split the time string and pad with zeros if parts are missing
        parts = (time_str.split(':') + ['00', '00'])[:3]
        hours, minutes, seconds = [int(x) for x in parts]
        return hours * 60 + minutes + seconds / 60
    except (ValueError, AttributeError):
        return 0.0

def process_time_columns(df):
    """
    Process time-related columns in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        pd.DataFrame: DataFrame with processed time columns
    """
    time_columns = [
        'Onset to door', 'Alert to door', 'Door to imaging',
        'Door to groin', 'Door to first series', 'Time of intervention',
        'Door to recanalization'
    ]
    
    for col in time_columns:
        if col in df.columns:
            df[f'{col} (minutes)'] = df[col].apply(convert_time_str_to_minutes)
    
    return df

def minutes_to_time_str(minutes):
    """
    Convert minutes to HH:MM:SS format.
    
    Args:
        minutes (float): Time in minutes
    
    Returns:
        str: Time in HH:MM:SS format
    """
    if pd.isna(minutes):
        return "00:00:00"
    
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    secs = int((minutes * 60) % 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"

def analyze_time_column(df, col_name, ax=None):
    """
    Perform detailed analysis of a time column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the time column
        col_name (str): Name of the time column (minutes version)
        ax (matplotlib.axes.Axes, optional): Axes object to plot on
    """
    # Get original column name (without ' (minutes)')
    orig_col = col_name.replace(' (minutes)', '')
    
    # Calculate statistics
    data = df[col_name]
    non_zero_data = data[data > 0]
    
    print(f"\n{'-' * 20} {orig_col} Analysis {'-' * 20}")
    
    # Basic statistics
    stats = {
        'Count (total)': len(data),
        'Count (non-zero)': len(non_zero_data),
        'Count (zero/missing)': len(data) - len(non_zero_data),
        'Mean': non_zero_data.mean(),
        'Median': non_zero_data.median(),
        'Std': non_zero_data.std(),
        'Min': non_zero_data.min(),
        'Max': non_zero_data.max(),
        'Q1 (25%)': non_zero_data.quantile(0.25),
        'Q3 (75%)': non_zero_data.quantile(0.75),
        'IQR': non_zero_data.quantile(0.75) - non_zero_data.quantile(0.25)
    }
    
    print("\nStatistics:")
    # Print count statistics
    for stat in ['Count (total)', 'Count (non-zero)', 'Count (zero/missing)']:
        print(f"{stat:20}: {stats[stat]}")
    
    # Print time statistics in both formats
    print("\nTime Statistics:")
    for stat in ['Mean', 'Median', 'Std', 'Min', 'Max', 'Q1 (25%)', 'Q3 (75%)', 'IQR']:
        if isinstance(stats[stat], float):
            time_str = minutes_to_time_str(stats[stat])
            print(f"{stat:20}: {time_str} ({stats[stat]:.2f} minutes)")
    
    # Calculate percentage of zero/missing values
    zero_pct = ((len(data) - len(non_zero_data)) / len(data)) * 100
    print(f"\nPercentage of zero/missing values: {zero_pct:.2f}%")
    
    # Create histogram if ax is provided
    if ax is not None:
        ax.hist(non_zero_data, bins=20, edgecolor='black')
        ax.set_title(f'Distribution of {orig_col}\n(non-zero values)')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Frequency')
        
        # Add vertical lines for key statistics
        ax.axvline(non_zero_data.median(), color='r', linestyle='dashed', linewidth=1, label='Median')
        ax.axvline(non_zero_data.quantile(0.25), color='g', linestyle='dashed', linewidth=1, label='Q1')
        ax.axvline(non_zero_data.quantile(0.75), color='g', linestyle='dashed', linewidth=1, label='Q3')
        
        ax.legend()

def analyze_demographics(df):
    """
    Perform basic demographic analysis on the dataset.
    
    Args:
        df (pd.DataFrame): Combined demographics DataFrame
    """
    # Process time columns first
    df = process_time_columns(df)
    
    print("\nDemographic Data Summary:")
    print("-" * 50)
    
    # Basic statistics for numerical columns
    print("\nNumerical Features Summary:")
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    # Exclude the processed time columns from the main numerical summary
    time_cols = [col for col in numerical_df.columns if '(minutes)' in col]
    non_time_cols = [col for col in numerical_df.columns if col not in time_cols]
    
    if non_time_cols:
        print("\nNon-time numerical features:")
        print(df[non_time_cols].describe())
    
    try:
        import matplotlib.pyplot as plt
        
        # Create a figure with subplots for all time columns
        n_cols = 3  # Number of columns in the subplot grid
        n_rows = (len(time_cols) + 1) // 2  # Number of rows needed
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()  # Flatten axes array for easier indexing
        
        # Detailed analysis of each time column
        for idx, time_col in enumerate(time_cols):
            analyze_time_column(df, time_col, ax=axes[idx])
        
        # Remove any unused subplots
        for idx in range(len(time_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        # Adjust layout and display all plots
        plt.tight_layout()
        
    except ImportError:
        print("\nNote: Matplotlib is not installed. Skipping histogram visualizations.")
        # If matplotlib is not available, still perform the numerical analysis
        for time_col in time_cols:
            analyze_time_column(df, time_col)
    
    # Distribution of categorical variables
    print("\nCategorical Features Distribution:")
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in categorical_cols:
        if col != 'id' and 'door' not in col.lower() and 'time' not in col.lower():
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False))
    
    # Missing values analysis
    print("\nMissing Values Analysis:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_pct
    })
    print(missing_info[missing_info['Missing Values'] > 0])
    
    # Show all plots at the end if matplotlib is available
    if 'plt' in locals():
        plt.show()
    
    # NIHSS Analysis
    print("\n" + "="*50)
    print("NIHSS Analysis")
    print("="*50)
    
    nihss_data = df['NIHSS at admission'].dropna()
    print(f"\nNumber of records with NIHSS scores: {len(nihss_data)}")
    
    # Basic statistics for NIHSS
    print("\nNIHSS Statistics:")
    stats = nihss_data.describe()
    for stat_name, value in stats.items():
        print(f"{stat_name:20}: {value:.2f}")
    
    # Create stratified folds
    fold_assignments, fold_stats = create_stratified_folds(df)
    
    # Print fold distributions
    print("\nFinal Fold Distributions:")
    print("-" * 40)
    for fold, stats in fold_stats['fold_distributions'].items():
        print(f"\nFold {fold}:")
        print("Severity distribution:")
        for severity, count in stats['severity_distribution'].items():
            print(f"{severity:20}: {count:3d} cases")
        print(f"Total cases: {stats['total_cases']}")
    
    analyze_fold_distributions(df, fold_assignments, plt)

def analyze_fold_distributions(df, fold_assignments, plt):
    """
    Analyze and visualize the distribution of variables across folds.
    """
    import seaborn as sns
    from scipy import stats
    
    # Select variables to analyze
    numeric_vars = ['Age', 'Glucose', 'Leucocytes', 'CRP', 'INR']
    categorical_vars = ['Sex', 'Atrial fibrillation', 'Hypertension', 'Diabetes', 'Hyperlipidemia']
    time_vars = ['Onset to door (minutes)', 'Door to imaging (minutes)']
    
    total_vars = len(numeric_vars) + len(categorical_vars) + len(time_vars)
    
    # Create a figure with subplots for each variable
    fig, axes = plt.subplots(total_vars, 1, figsize=(15, 4*total_vars))
    fig.suptitle('Distribution of Variables Across Folds', fontsize=16, y=0.95)
    
    current_ax = 0
    
    # Function to create boxplots for numeric variables
    def plot_numeric_var(var_name, ax_idx):
        data_to_plot = []
        labels = []
        for fold in range(5):
            fold_data = df[df.index.isin(fold_assignments[fold])][var_name].dropna()
            data_to_plot.append(fold_data)
            labels.append(f'Fold {fold}')
        
        axes[ax_idx].boxplot(data_to_plot, labels=labels)
        axes[ax_idx].set_title(f'{var_name} Distribution')
        axes[ax_idx].set_ylabel(var_name)
        
        # Add mean values as text
        means = [data.mean() for data in data_to_plot]
        for i, mean in enumerate(means):
            axes[ax_idx].text(i+1, axes[ax_idx].get_ylim()[0], 
                            f'μ={mean:.1f}', 
                            horizontalalignment='center',
                            verticalalignment='top')
    
    # Function to create bar plots for categorical variables
    def plot_categorical_var(var_name, ax_idx):
        fold_proportions = []
        for fold in range(5):
            fold_data = df[df.index.isin(fold_assignments[fold])][var_name]
            props = fold_data.value_counts(normalize=True)
            fold_proportions.append(props)
        
        # Convert to DataFrame for easier plotting
        prop_df = pd.DataFrame(fold_proportions).fillna(0)
        prop_df.plot(kind='bar', ax=axes[ax_idx])
        axes[ax_idx].set_title(f'{var_name} Distribution')
        axes[ax_idx].set_xlabel('Fold')
        axes[ax_idx].set_ylabel('Proportion')
        axes[ax_idx].legend(title='Value')
    
    # Plot numeric variables
    print("\nAnalyzing numeric variables...")
    for var in numeric_vars:
        plot_numeric_var(var, current_ax)
        current_ax += 1
    
    # Plot categorical variables
    print("Analyzing categorical variables...")
    for var in categorical_vars:
        plot_categorical_var(var, current_ax)
        current_ax += 1
    
    # Plot time variables
    print("Analyzing time variables...")
    for var in time_vars:
        plot_numeric_var(var, current_ax)
        current_ax += 1
    
    plt.tight_layout()
    plt.show()
    
    # Print statistical tests
    print("\nStatistical Analysis of Fold Distributions:")
    print("-" * 50)
    
    for var in numeric_vars + time_vars:
        print(f"\n{var}:")
        try:
            # Prepare data for ANOVA
            fold_data = [df[df.index.isin(fold_assignments[fold])][var].dropna() 
                       for fold in range(5)]
            
            # Perform one-way ANOVA
            f_stat, p_value = stats.f_oneway(*fold_data)
            print(f"One-way ANOVA: F={f_stat:.2f}, p={p_value:.4f}")
        except:
            print("Could not perform statistical test due to insufficient data")
    
    for var in categorical_vars:
        print(f"\n{var}:")
        try:
            # Create contingency table
            contingency = pd.DataFrame([
                df[df.index.isin(fold_assignments[fold])][var].value_counts()
                for fold in range(5)
            ])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            print(f"Chi-square test: χ²={chi2:.2f}, p={p_value:.4f}")
        except:
            print("Could not perform statistical test due to insufficient data")

def create_stratified_folds(df, nihss_column='NIHSS at admission', id_column=None):
    """
    Create 5 stratified folds based on NIHSS severity categories.
    
    Args:
        df (pd.DataFrame): DataFrame containing the NIHSS scores
        nihss_column (str): Name of the column containing NIHSS scores. Default 'NIHSS at admission'
        id_column (str): Optional name of ID column. If None, uses DataFrame index
    
    Returns:
        dict: Dictionary mapping fold numbers (0-4) to lists of case IDs
        dict: Dictionary containing fold statistics and distributions
    
    Example:
        fold_assignments, fold_stats = create_stratified_folds(df)
        # Access cases for fold 0
        fold_0_cases = fold_assignments[0]
        # Access statistics
        print(fold_stats['distributions'])
    """
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
    
    # Create a DataFrame with case IDs and their severity labels
    stratification_df = pd.DataFrame({
        'id': df.index if id_column is None else df[id_column],
        'NIHSS': df[nihss_column],
        'severity': df[nihss_column].apply(get_severity_label)
    }).dropna()
    
    # Initialize fold assignments and statistics
    fold_assignments = {0: [], 1: [], 2: [], 3: [], 4: []}
    fold_stats = {
        'total_cases': len(stratification_df),
        'cases_per_fold': len(stratification_df) / 5,
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
            'ideal_per_fold': cases_in_category / 5
        }
        
        # Distribute cases across folds
        for idx, case in enumerate(severity_cases.itertuples()):
            fold = idx % 5
            fold_assignments[fold].append(case.id)
    
    # Calculate final distributions for each fold
    for fold in range(5):
        fold_cases = stratification_df[stratification_df['id'].isin(fold_assignments[fold])]
        severity_counts = fold_cases['severity'].value_counts().to_dict()
        fold_stats['fold_distributions'][fold] = {
            'total_cases': len(fold_cases),
            'severity_distribution': severity_counts
        }
    
    return fold_assignments, fold_stats

def main():
    parser = argparse.ArgumentParser(description="Analyze demographics data from the dataset.")
    parser.add_argument("dataset_path", help="Path to the dataset root directory")
    parser.add_argument("--output", help="Path to save the combined demographics CSV file (optional)")
    args = parser.parse_args()
    
    try:
        # Load all demographics data
        print("Loading demographics data...")
        df = load_demographics_data(args.dataset_path)
        
        # Print basic information about the dataset
        print(f"\nLoaded {len(df)} cases")
        print(f"Features available: {', '.join(df.columns.tolist())}")
        
        # Perform demographic analysis
        analyze_demographics(df)
        
        # Save combined data if output path is provided
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nCombined demographics data saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 