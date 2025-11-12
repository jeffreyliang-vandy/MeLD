import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import scipy.stats as stats


### Bin Countinous
def bin_numeric(vector,bins=50,mask_na = True):
    # Create a NA mask
    na_mask = np.isnan(vector).astype(int)

    # Bin continous vector
    vector = np.array(vector).astype(float)
    bins = np.min([bins,len(np.unique(vector))-1])
    bin_index = pd.qcut(vector,bins,duplicates='drop')
    bin_matrix = pd.get_dummies(bin_index,drop_first=False).astype(int)
    
    # Combined matrix
    if all([mask_na,sum(na_mask) >0]):
        bin_matrix["nan"] = na_mask

    return bin_matrix*1

# bin_numeric([1,2,2,3,4,np.nan,5],bins=10)


### bin date

def bin_date(date):
    date = date-1900
    ### 100 
    year_100 = date//100
    year_10 = date//10 - year_100*10
    year_1 = date % 10
    month_date = 10*(date%1)//1
    bin_year = np.zeros(32).astype(int)
    if np.isnan(date): return bin_year
    bin_year[np.array([year_100,year_10+2,year_1+12,month_date+22]).astype(int)]=1

    return bin_year

# bin_date(np.nan)

### Numeric Date:
def datetime_to_continuous_year(date_str):
    if pd.isna(date_str) or date_str is None:
        # return 1800  # Return a very large number for NA date-time
        return np.nan
    
    # Convert the date-time string to a datetime object
    date_obj = pd.Timestamp(date_str)
    
    # Calculate the start of the year and the next year for the given date
    start_of_year = pd.Timestamp(date_obj.year, 1, 1)
    start_of_next_year = pd.Timestamp(date_obj.year + 1, 1, 1)
    
    # Calculate the fraction of the year that has passed
    year_fraction = (date_obj - start_of_year) / (start_of_next_year - start_of_year)
    
    # Return the continuous year
    if date_obj.year <= 1900: return np.nan
    return date_obj.year + year_fraction

# Test
# print(datetime_to_continuous_year("1900-1-2"))  # Expected output: 2021.5 (or something close)
# print(datetime_to_continuous_year(None))  # Expected output: 9999

### bin method from https://www.nature.com/articles/s41746-023-00888-7#Sec11

def normalize_numeric(vector):
    vector = np.array(vector).astype(float)
    uniq_x = np.unique(vector)
    N = len(vector)

    lower_b = 0
    upper_b = 0

    stochastic_x = vector.copy()
    params = dict()

    for val in uniq_x:
        idx = np.where(vector == val)[0]
        ratio_val = len(idx)/N
        upper_b = lower_b + ratio_val
        stochastic_x[idx] = np.random.uniform(low=lower_b,high=upper_b,size=len(idx))
        params[val] = [lower_b,upper_b]
        lower_b = upper_b

    return({'x':stochastic_x,'param':params})

### Mask Missing data
def find_missing_index(vector):
    return pd.isnull(vector)

# find_missing_index(['a', np.nan, 'are'])

### Categorical dtype 
def ascat(x):
    x = x.apply(lambda x: str(x).strip() if pd.notna(x) else x)
    
    # # combine the last category a with the second last b as a+b, until the last category is greater than 0.01
    # cat_prob = x.value_counts(normalize=True)
    # while cat_prob.iloc[-1] < 0.01:
    #     combined_cat = cat_prob.index[-1] + '+' + cat_prob.index[-2]
    #     x = x.replace(cat_prob.index[-1], combined_cat)
    #     x = x.replace(cat_prob.index[-2], combined_cat)
    #     cat_prob = x.value_counts(normalize=True)

    categories = x.dropna().astype(str).unique()  # Get unique values excluding NaN
    cat_type = pd.api.types.CategoricalDtype(categories=categories, ordered=False)
    x = x.astype(cat_type)
    return x




def plot_binned_histogram(df, variable_prefix, bin_width=10.0, x_transform=None, ax=None):
    """
    Plot a histogram for binned continuous variables in a DataFrame with optional x-axis transformation using seaborn.
    
    Args:
    df (pd.DataFrame): The input DataFrame.
    variable_prefix (str): The prefix of the binned variables (e.g., 'weight' or 'height').
    bin_width (float, optional): The width of the bins for the histogram. Defaults to 10.0.
    x_transform (callable, optional): A function to transform the x-axis values. Defaults to None.
    ax (matplotlib.axes.Axes, optional): The axes on which to draw the plot. If not provided, a new figure will be created.
    
    Returns:
    None: Displays the histogram.
    """
    
    # Filter columns that match the variable prefix (e.g., 'weight_')
    variable_columns = df.filter(regex=f'^{variable_prefix}_').columns

    # Initialize lists to store midpoints and their corresponding counts
    bin_midpoints = []
    bin_counts = []

    # Loop through columns that match the variable prefix
    for col in variable_columns:
        # Extract bin ranges using regex
        match = re.search(rf'{variable_prefix}_\(([\d.]+),\s*([\d.]+)\]', col)
        if match:
            lower_bound = float(match.group(1))
            upper_bound = float(match.group(2))
            
            # Calculate the midpoint of the bin
            midpoint = (lower_bound + upper_bound) / 2
            bin_midpoints.append(midpoint)
            
            # Get the count of non-null values in the column (assuming 1 indicates presence)
            count = df[col].sum()
            bin_counts.append(count)
    
    # Apply the x_transform function if provided, otherwise leave midpoints as is
    if x_transform:
        bin_midpoints = list(map(x_transform, bin_midpoints))

    # Creating a DataFrame with midpoints and counts for seaborn compatibility
    data = pd.DataFrame({
        'midpoints': bin_midpoints,
        'counts': bin_counts
    })

    # If no axis (ax) is provided, create a new figure and axis
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    # Create the histogram plot using seaborn on the provided or newly created axis
    sns.histplot(data=data, x='midpoints', weights='counts', binwidth=bin_width, kde=False, edgecolor='black', ax=ax)
    
    # Customizing labels
    ax.set_xlabel(f'{variable_prefix.capitalize()} (Transformed)' if x_transform else f'{variable_prefix.capitalize()} (Units)')
    ax.set_ylabel('Frequency')
    # ax.set_title(f'Histogram of {variable_prefix.capitalize()} Distribution with Bin Width {bin_width}')
    ax.grid(True)


def calculate_moments(samples):
    """
    Function to calculate the first four moments: mean, variance, skewness, and kurtosis.
    """
    mean = np.mean(samples).astype(float)
    variance = np.var(samples, ddof=1).astype(float)
    skewness = stats.skew(samples).astype(float)
    # kurtosis = stats.kurtosis(samples, fisher=False)
    upper = np.nanmax(samples).astype(float)
    lower = np.nanmin(samples).astype(float)
    
    # return mean, variance, skewness, kurtosis
    recur = lambda x: float(x)
    v = [skewness, mean, np.sqrt(variance),lower, upper]
    v = tuple(map(recur,v))
    return v

def create_one_hot_mapping(original_columns, one_hot_columns):
    # Initialize a dictionary to hold the mappings
    one_hot_mapping = {}
    
    # Iterate over each original column
    for col in original_columns:
        # For each original column, find all corresponding one-hot columns
        one_hot_mapping[col] = [one_hot_col for one_hot_col in one_hot_columns if one_hot_col.startswith(f"{col}_")]
        if len(one_hot_mapping[col])==0:
            one_hot_mapping[col] = [col]
    
    return one_hot_mapping

# # Example usage:
# original_columns = ['A', 'B',"C"]
# one_hot_columns = ['A_a', 'A_b', 'B_(0,3]', 'B_nan',"D"]

# # Create the mapping
# mapping = create_one_hot_mapping(original_columns, one_hot_columns)
# print(mapping)