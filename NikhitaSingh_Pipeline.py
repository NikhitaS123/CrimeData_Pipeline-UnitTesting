import os
import pandas as pd
from scipy.spatial import KDTree
import logging

LOCAL_DATA_PATH = './'
LOG_FILE = os.path.join(LOCAL_DATA_PATH, 'pipeline.log')
STAGED_CRIME= os.path.join(LOCAL_DATA_PATH, 'staged_crime.csv')
STAGED_ETHNICITY = os.path.join(LOCAL_DATA_PATH, 'staged_ethnicity.csv')
STAGED_POSTCODE = os.path.join(LOCAL_DATA_PATH, 'staged_postcode.csv')
PRIMARY_CRIME_FILE = os.path.join(LOCAL_DATA_PATH, 'primary_crime.csv')
REPORTING_CRIME_FILE = os.path.join(LOCAL_DATA_PATH, 'reporting_crime.csv')

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def ingest_data(file_path: str)-> pd.DataFrame:
    """
    Ingest raw data from a CSV file. Pass in the file path as a string and returns a pandas dataframe.
    """
    logging.info(f"Starting data ingestion from {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data ingestion from {file_path} completed successfully")
        return df
    except Exception as e:
        logging.error(f"Error reading the CSV file {file_path}: {e}")
        raise ValueError(f"Error reading the CSV file {file_path}: {e}")
        
""" read files and creating dataframe function"""

def concat_files(main_folder_path, file_extension='.csv',excel_extension='.xlsx', sheet_name=None):
    """
    Reads and concatenates all files with a given extension from subfolders within a main folder into a single DataFrame.

    Parameters:
    main_folder_path (str): The path to the main folder containing subfolders with files.
    file_extension (str): The extension of the files to read (default is '.csv').

    Returns:
    pd.DataFrame: A DataFrame containing the combined data from all files in the subfolders.
    """
    # Initialize an empty list to store DataFrames
    dataframes = []

    # Walk through all subfolders in the main folder
    for root, dirs, files in os.walk(main_folder_path):
        for filename in files:
            # Check if the file matches the desired extension
            if filename.endswith(file_extension):
                # Construct full file path
                file_path = os.path.join(root, filename)
                #print(f"Reading file: {file_path}")
                
                # Log the file being read
                logging.info(f"Reading file: {file_path}")
                try:
                    # Read the file into a DataFrame and append it to the list
                    if file_extension == '.csv':
                        logging.info(f"Reading CSV file: {file_path}")
                        print(f"Reading CSV file: {file_path}")
                        df = pd.read_csv(file_path)
                        dataframes.append(df)
                    elif file_extension == '.xlsx':
                        logging.info(f"Reading Excel file: {file_path}, sheet: {sheet_name}")
                        print(f"Reading Excel file: {file_path}, sheet: {sheet_name}")
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        dataframes.append(df)
                    else:
                        raise ValueError(f"Unsupported file extension: {file_extension}")
                except Exception as e:
                    logging.error(f"Error reading file {file_path}: {str(e)}")
                    continue  # Skip the problematic file and continue with the next one
                dataframes.append(df)
    
    if not dataframes:
        error_message = "No files with the specified extension were found in the subfolders."
        logging.error(error_message)
        raise ValueError("No files with the specified extension were found in the subfolders.")

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

     # Log successful concatenation
    logging.info(f"Successfully concatenated {len(dataframes)} files into a single DataFrame.")

    return combined_df

def outer_join_on_index(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Performs an outer join on two DataFrames based on their indices.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.

    Returns:
    pd.DataFrame: The DataFrame resulting from the outer join.
    """

    # Log the start of the function
    logging.info("Starting outer join on DataFrames based on index.")
    
    # Log DataFrame details
    logging.info(f"DataFrame 1 columns: {df1.columns.tolist()}")
    logging.info(f"DataFrame 2 columns: {df2.columns.tolist()}")
    
    # Perform an outer join on index
    df_combined = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')

    # Log the result of the join
    logging.info(f"Combined DataFrame shape: {df_combined.shape}")
    logging.info("Outer join completed successfully.")

    return df_combined

def delete_except(df, keep_columns):
    """
    Delete all columns in the DataFrame except for the specified ones.
    
    Parameters:
    - df: The DataFrame from which columns are to be deleted.
    - keep_columns: A list of column names that you want to keep.
    
    Returns:
    - The modified DataFrame with only the columns you want to keep.
    """
    columns_to_drop = [col for col in df.columns if col not in keep_columns]
    
     # Log the columns that will be kept and dropped
    logging.info(f"Keeping columns: {keep_columns}")
    logging.info(f"Dropping columns: {columns_to_drop}")

    try:
    # Drop the columns that are not in the keep list
        df.drop(columns=columns_to_drop, inplace=True)
        logging.info("Columns successfully dropped.")
    except KeyError as e:
        logging.error(f"Error: {str(e)}")
    return df

def filter_func(df, column_name, words):
    """
    Filters a DataFrame based on the presence of multiple words in a column, 
    even if other words surround them (no word boundaries enforced).
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be filtered.
    column_name (str): The column in which to search for the words.
    words (list of str): A list of words to filter by.
    
    Returns:
    pd.DataFrame: A filtered DataFrame containing rows where any of the words were found.
    """
    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist in the DataFrame.")
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

     # Log the words being used for filtering
    logging.info(f"Filtering by words: {words} in column: {column_name}")
    
    # Create a regex pattern to match any of the words
    pattern = '|'.join(words)  # No \b here to allow partial matches
    logging.info(f"Regex pattern created: {pattern}")
    
    # Filter the DataFrame based on the pattern match
    filtered_df = df[df[column_name].str.contains(pattern, case=False, na=False)]
    # Log the number of rows in the filtered DataFrame
    logging.info(f"Number of rows after filtering: {len(filtered_df)}")
    
    return filtered_df

def splitting_LSOAname(df, column_name):
    """
    Splits a column by the last four characters and creates two new columns.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the column.
    column_name (str): The name of the column to be split.
    
    Returns:
    pd.DataFrame: The DataFrame with two new columns: 'Before_Last_Four' and 'Last_Four'.
    """
    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist in the DataFrame.")
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Log the start of the column splitting process
    logging.info(f"Starting to split the column: {column_name}")
    
    # Ensure the column is treated as a string
    df[column_name] = df[column_name].astype(str)
    logging.info(f"Column '{column_name}' converted to string data type.")
    
    # Create two new columns: one for the part before the last four digits and one for the last four digits
    df['LSOA firstname'] = df[column_name].str[:-4]
    df['LSOA namecode'] = df[column_name].str[-4:]

    # Log the creation of new columns
    logging.info(f"New columns 'LSOA firstname' and 'LSOA namecode' created from '{column_name}'.")
    
    return df

def reset_index(df, drop=True, inplace=False):
    """
    Resets the index of the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame whose index will be reset.
    drop (bool): Whether to drop the old index. Default is True.
    inplace (bool): Whether to modify the DataFrame in place. Default is False.
    
    Returns:
    pd.DataFrame: DataFrame with the index reset (unless inplace=True).
    """
     # Log the parameters for transparency
    logging.info(f"Resetting index for DataFrame. Drop: {drop}, Inplace: {inplace}")
    
    # Check if inplace is set to True
    if inplace:
        logging.info("Resetting index inplace.")
        df.reset_index(drop=drop, inplace=True)
    else:
        logging.info("Resetting index and returning a new DataFrame.")
        return df.reset_index(drop=drop)

def delete_columns(df, columns_to_drop):
    """
    Deletes specified columns from the DataFrame in-place.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    columns_to_drop (str or list of str): Column name or list of column names to drop.
    
    Returns:
    None: The DataFrame is modified in-place.
    """
    # Ensure columns_to_drop is a list
    if isinstance(columns_to_drop, str):
        columns_to_drop = [columns_to_drop]
        # Log which columns are being dropped
    logging.info(f"Attempting to drop columns: {columns_to_drop}")
    try:
        # Drop columns in-place
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        logging.info(f"Successfully dropped columns: {columns_to_drop}")
    except Exception as e:
        logging.error(f"Failed to drop columns: {columns_to_drop}.Error:{str(e)}")

def remove_nan(df, column_name):
    """
    Removes rows with NaN values from a specified column in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the column.
    column_name (str): The name of the column from which NaN values should be removed.
    
    Returns:
    pd.DataFrame: The DataFrame with NaN values removed from the specified column.
    """
     # Log the column check
    logging.info(f"Attempting to remove NaN values from column: {column_name}")
    
    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist in the DataFrame.")
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        
    # Log before removing NaN values
    initial_row_count = df.shape[0]
    logging.info(f"Initial row count: {initial_row_count}")
    
    # Drop rows where the specified column has NaN values
    df_cleaned = df.dropna(subset=[column_name])

    # Log the number of rows after cleaning
    final_row_count = len(df_cleaned)
    logging.info(f"Number of rows after removing NaN values: {final_row_count}")
    
    logging.info(f"Final row count after removing NaN values: {final_row_count} (Removed {initial_row_count - final_row_count} rows)")
    
    return df_cleaned

def fill_nan(df):
    """
    Fills NaN values in all columns of a DataFrame with 'No(column_name)'.
    
    Parameters:
    df (pd.DataFrame): The DataFrame in which NaN values will be filled.
    
    Returns:
    pd.DataFrame: The DataFrame with NaN values replaced by 'No(column_name)'.
    """
     # Log the initial state of the DataFrame
    logging.info("Starting to fill NaN values with 'No(column_name)' placeholders.")
    logging.info(f"Initial number of NaN values: {df.isna().sum().sum()}")

    for column in df.columns:
        # Create the placeholder string for each column
        placeholder = f"No {column}"
        nan_count_before = df[column].isna().sum()

        # Log the NaN count for the current column before filling
        logging.info(f"Filling NaN values in column: '{column}' (NaN count: {nan_count_before})")
        
        # Fill NaN values with the placeholder in the column
        df[column] = df[column].fillna(placeholder)
        # Log the action performed
        logging.info(f"NaN values in column: '{column}' filled with '{placeholder}'")
        
     # Log the final state of the DataFrame
    logging.info(f"Final number of NaN values after filling: {df.isna().sum().sum()}")
    
    
    return df

def split_month(df, column_name):
    """
    Splits a column containing year and month (formatted as 'YYYY-MM') into two new columns: 'Year' and 'Month'.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to be split.
    column_name (str): The name of the column to split.
    
    Returns:
    pd.DataFrame: The updated DataFrame with new 'Year' and 'Month' columns.
    """
    # Check if the column exists in the DataFrame
    if column_name in df.columns:
        # Split the specified column and create 'Year' and 'Month' columns
        df[['Year', 'Month']] = df[column_name].str.split('-', expand=True)
    else:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    return df

def clean_columns(df):
    """
    Cleans all columns of object data type by removing leading/trailing spaces and converting to lowercase.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be cleaned.
    
    Returns:
    pd.DataFrame: The DataFrame with cleaned object-type columns.
    """
    # Log the start of the function
    logging.info("Starting to clean object columns.")

    
    # Loop through columns with 'object' dtype
    for col in df.select_dtypes(include=['object']).columns:
        # Log the column being processed
        logging.info(f"Cleaning column: {col}")
        # Apply strip and lower only on non-null values
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Log completion of the function
    logging.info("Finished cleaning object columns.")
    
    return df

def postcode_func(crime_df, postcode_df):
    # Log the start of the function
    logging.info("Starting postcode_func to find closest postcodes.")
    
    # Extract coordinates from the DataFrames
    postcode_coords = postcode_df[['Latitude', 'Longitude']].values
    crime_coords = crime_df[['Latitude', 'Longitude']].values

     # Log the number of coordinates
    logging.info(f"Number of postcode coordinates: {len(postcode_coords)}")
    logging.info(f"Number of crime coordinates: {len(crime_coords)}")

    # Build KDTree for postcode coordinates
    tree = KDTree(postcode_coords)
    logging.info("KDTree built for postcode coordinates.")
    
    # Query the KDTree to find the closest postcodes
    distances, indices = tree.query(crime_coords, k=1)  # k=1 for the closest point
    logging.info("KDTree query completed.")
    
    # Safely access postcodes using indices
    closest_postcodes = postcode_df.iloc[indices.flatten()]['Postcode'].values
    crime_df['Closest Postcode'] = closest_postcodes

    # Log the first few matched postcodes for verification
    logging.info(f"First few matched postcodes: {closest_postcodes[:5]}")
    
    # Calculate percentage of matched records if needed
    total_records = len(crime_df)
    matched_records = total_records  # Assuming all records are matched
    
    if total_records > 0:
        matched_percentage = (matched_records / total_records) * 100
    else:
        matched_percentage = 0
    
    print(f"Matched records percentage: {matched_percentage}%")
    logging.info(f"Matched records percentage: {matched_percentage}%")
    
    return crime_df

def staging():
    """
    Ingest the data, apply cleaning, and store to CSV files for staging.
    """
    logging.info("Starting Staging Layer")
    main_folder_path = 'JupyterNotebook/Crime2'
    # Call the function to read and concatenate all CSV files from subfolders
    crime_df = concat_files(main_folder_path, file_extension='.csv')
    # Display the combined DataFrame
    crime_df.head()
    main_folder_path = 'JupyterNotebook/Ethnicity'
    # Call the function to read and concatenate all CSV files from subfolders
    ethnicity_df = concat_files(main_folder_path, file_extension='.xlsx', sheet_name=3)
    # Display the combined DataFrame
    ethnicity_df.head()
    main_folder_path = 'JupyterNotebook/Postcode'
    # Call the function to read and concatenate all CSV files from subfolders
    postcode_df = concat_files(main_folder_path, file_extension='.csv')
    # Display the combined DataFrame
    postcode_df.head()
    
    try:
        # Apply transformations
        #Postcode Dataframe
        postcode_df=delete_except(postcode_df,['Postcode','Longitude','Latitude'])
        postcode_df=clean_columns(postcode_df)
        #Ethnicity Dataframe
        words_to_filter=['Pembrokeshire','Eastbourne']
        ethnicity_df=filter_func(ethnicity_df,'ConstituencyName',words_to_filter)
        reset_index(ethnicity_df,inplace=True)
        ethnicity_df=clean_columns(ethnicity_df)
        #Crime Dataframe
        crime_df = split_month(crime_df,'Month')
        crime_df=splitting_LSOAname(crime_df,'LSOA name')
        crime_df=filter_func(crime_df,'LSOA firstname',words_to_filter)
        reset_index(crime_df,inplace=True)
        columns_to_drop=['Context']
        delete_columns(crime_df,columns_to_drop)
        crime_df=remove_nan(crime_df,'Longitude')
        crime_df=remove_nan(crime_df,'Latitude')
        crime_df=fill_nan(crime_df)
        crime_df=clean_columns(crime_df)
        # Save staging files to CSV
        crime_df.to_csv(STAGED_CRIME, index=False)
        ethnicity_df.to_csv(STAGED_ETHNICITY, index=False)
        postcode_df.to_csv(STAGED_POSTCODE, index=False)
        logging.info("Data staging completed successfully")
    except Exception as e:
        logging.error(f"Error during data staging: {e}")

def primary():
    """
    Primary Layer: Store the transformed data to a CSV file.
    """
    logging.info("Starting Primary Layer")
     # ingest staging
    crime_df = ingest_data(STAGED_CRIME)
    ethnicity_df = ingest_data(STAGED_ETHNICITY)
    postcode_df = ingest_data(STAGED_POSTCODE)
    try:
        #adding postcode to crime df
        crime_df=postcode_func(crime_df,postcode_df)
        # merge 
        crime_df = outer_join_on_index(crime_df, ethnicity_df)
        # Save to CSV
        crime_df.to_csv(PRIMARY_CRIME_FILE, index=False)
        logging.info("Primary Layer completed successfully")
    except Exception as e:
        logging.error(f"Error during Primary Layer: {e}")

def reporting():
    """
    Reporting Layer: Store the aggregated reporting data to a CSV file.
    """
    logging.info("Starting Reporting Layer.")
    crime_postcode_ethnicity_df = ingest_data(PRIMARY_CRIME_FILE)
    try:
        # Apply aggregation directly within the function
        crime_outcome_df = crime_postcode_ethnicity_df.groupby(['Crime type', 'Outcome type']).size().reset_index(name='Count')
        crime_LSOAfirstname_df = crime_postcode_ethnicity_df.groupby(['Crime type', 'LSOA firstname']).size().reset_index(name='Count')
        crime_ethnicity_df = crime_postcode_ethnicity_df.groupby(['Crime type', 'groups']).size().reset_index(name='Count')
        LSOAfirstname_ethnicity_df = crime_postcode_ethnicity_df.groupby(['LSOA firstname', 'groups']).size().reset_index(name='Count')
        outcome_ethnicity_df = crime_postcode_ethnicity_df.groupby(['Outcome type', 'groups']).size().reset_index(name='Count')
        # Save to CSV
        crime_outcome_df.to_csv(REPORTING_CRIME_FILE, index=False)
        crime_LSOAfirstname_df.to_csv(REPORTING_CRIME_FILE, index=False)
        crime_ethnicity_df.to_csv(REPORTING_CRIME_FILE, index=False)
        LSOAfirstname_ethnicity_df.to_csv(REPORTING_CRIME_FILE, index=False)
        outcome_ethnicity_df.to_csv(REPORTING_CRIME_FILE, index=False)
        logging.info("Reporting data completed successfully")
    except Exception as e:
        logging.error(f"Error during reporting data: {e}")

def main(pipeline='all'):
    logging.info("Pipeline execution started")

    try:
        if pipeline in ['all', 'staging', 'primary', 'reporting']:
            staging()
            logging.info("Staging execution completed successfully")
            if pipeline == 'staging':
                # If only staging is requested, print success and return
                logging.info("Pipeline run complete")
                return
            # Process the staged data
            primary()
            logging.info("Primary execution completed successfully")
            if pipeline == 'primary':
                # If only primary is requested, print success and return 
                logging.info("Pipeline run complete")
                return
            # Generate reports based on processed data
            reporting()
            logging.info("Reporting execution completed successfully")
            if pipeline == 'reporting':
                logging.info("Pipeline run complete")
                return
            logging.info("Full pipeline run complete")
        else:
            # Inform the user about an invalid pipeline stage input
            logging.critical("Invalid pipeline stage specified. Please choose 'staging', 'primary', 'reporting', or 'all'.")
    except Exception as e:
        # Catch and print any exceptions occurred during pipeline execution
        logging.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()