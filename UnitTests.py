import unittest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from io import StringIO
import logging
import pytest
from unittest.mock import patch


# Import the functions you want to test
#from NikhitaSingh_Pipeline import ingest_data, outer_join_on_index, delete_except, filter_func, splitting_LSOAname, concat_files
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


"""Testing STARTS HERE"""
class TestPipelineFunctions(unittest.TestCase):

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_ingest_data_success(self, mock_read_csv, mock_exists):
        mock_exists.return_value = True
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_csv.return_value = mock_df
        
        result = ingest_data('test.csv')
        self.assertTrue(mock_read_csv.called)
        self.assertEqual(result.shape, (2, 2))

    @patch('os.path.exists')
    def test_ingest_data_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            ingest_data('nonexistent.csv')

    def test_outer_join_on_index(self):
        df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
        df2 = pd.DataFrame({'B': [3, 4]}, index=[0, 1])
        result = outer_join_on_index(df1, df2)
        self.assertEqual(result.shape, (2, 2))
        self.assertIn('A', result.columns)
        self.assertIn('B', result.columns)

    def test_delete_except(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        result = delete_except(df, ['A'])
        self.assertEqual(list(result.columns), ['A'])

    def test_filter_func(self):
        df = pd.DataFrame({'col1': ['This is a test', 'Another test', 'No match here']})
        words = ['test']
        result = filter_func(df, 'col1', words)
        self.assertEqual(len(result), 2)
        self.assertIn('This is a test', result['col1'].values)

    def test_splitting_LSOAname(self):
        df = pd.DataFrame({'LSOA name': ['abcd1234', 'efgh5678']})
        result = splitting_LSOAname(df, 'LSOA name')
        self.assertEqual(result['LSOA firstname'].tolist(), ['abcd', 'efgh'])
        self.assertEqual(result['LSOA namecode'].tolist(), ['1234', '5678'])

    @patch('os.walk')
    def test_concat_files(self, mock_walk):
        # Mock the os.walk response to simulate files in subfolders
        mock_walk.return_value = [
            ('root', ['subfolder1'], ['file1.csv', 'file2.csv']),
            ('root/subfolder1', [], ['file3.csv']),
        ]

        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                pd.DataFrame({'A': [1]}),
                pd.DataFrame({'A': [2]}),
                pd.DataFrame({'A': [3]})
            ]
            
            result = concat_files('test_path')
            self.assertEqual(result.shape[0], 3)  # 3 rows from 3 files

if __name__ == '__main__':
    unittest.main()

import pytest
import pandas as pd

# 1. Create a fixture to provide test data
@pytest.fixture
def test_data():
    # Create a simple DataFrame with a 'Date' column
    data = {'Date': ['2021-01', '2022-05', '2023-12']}
    df = pd.DataFrame(data)
    return df

# 2. Test the valid case
def test_split_month_valid(test_data):
    df = test_data
    # Call the function to split 'Date' into 'Year' and 'Month'
    result = split_month(df, 'Date')
    
    # Check if the 'Year' and 'Month' columns were created correctly
    assert 'Year' in result.columns
    assert 'Month' in result.columns
    assert result['Year'].tolist() == ['2021', '2022', '2023']
    assert result['Month'].tolist() == ['01', '05', '12']

# 3. Test when the column does not exist
def test_split_month_column_not_found(test_data):
    df = test_data
    # Call the function with a non-existing column and check for ValueError
    with pytest.raises(ValueError, match="Column 'NonExistent' not found in DataFrame"):
        split_month(df, 'NonExistent')

import pytest
import pandas as pd

# 1. Create a fixture for sample DataFrame
@pytest.fixture
def sample_df():
    data = {
        'Text': [
            'The quick brown fox',
            'Jumps over the lazy dog',
            'Python is great for data analysis',
            'Pandas is a useful library',
            'Data science is fun'
        ]
    }
    return pd.DataFrame(data)

# 2. Test case for valid filtering
def test_filter_func_valid(sample_df):
    df = sample_df
    # Call filter_func with a valid column and word list
    filtered_df = filter_func(df, 'Text', ['Python', 'Pandas', 'fox'])
    
    # Check if the correct rows are returned (case-insensitive matches)
    assert len(filtered_df) == 3  # 'fox', 'Python', 'Pandas' should match
    assert 'The quick brown fox' in filtered_df['Text'].values
    assert 'Python is great for data analysis' in filtered_df['Text'].values
    assert 'Pandas is a useful library' in filtered_df['Text'].values

# 3. Test case for column not found
def test_filter_func_column_not_found(sample_df):
    df = sample_df
    with pytest.raises(ValueError, match="Column 'NonExistent' does not exist in the DataFrame"):
        filter_func(df, 'NonExistent', ['Python', 'Pandas'])

# 4. Test case for no matching words
def test_filter_func_no_matches(sample_df):
    df = sample_df
    # Call filter_func with words that are not in the 'Text' column
    filtered_df = filter_func(df, 'Text', ['unicorn', 'dragons'])
    
    # Check if no rows are returned
    assert len(filtered_df) == 0  # No matches should result in an empty DataFrame

# 5. Test case for empty words list
def test_filter_func_empty_words_list(sample_df):
    df = sample_df
    # Call filter_func with an empty list of words
    filtered_df = filter_func(df, 'Text', [])
    
    # Check if no rows are returned since there are no words to match
    assert len(filtered_df) == 0  # An empty list should return an empty DataFrame


@pytest.fixture
def sample_df():
    data = {
        'Name': [' John ', ' Jane ', 'DOE '],
        'City': [' New York ', 'London', ' PARIS '],
        'Age': [25, 30, 22]  # This is an int column, should not be affected
    }
    return pd.DataFrame(data)

# Test cleaning functionality with mocked logging
@patch('your_module_name.logging.info')  # Mock the logging.info method
def test_clean_columns(mock_logging_info, sample_df):
    df = sample_df
    
    # Call the clean_columns function
    cleaned_df = clean_columns(df)
    
    # Assert that logging.info was called (log cleaning start, per column, and finish)
    assert mock_logging_info.call_count == 5  # 1 for start, 3 columns, 1 for finish

    # Assert specific log messages (for example)
    mock_logging_info.assert_any_call("Starting to clean object columns.")
    mock_logging_info.assert_any_call("Cleaning column: Name")
    mock_logging_info.assert_any_call("Cleaning column: City")
    mock_logging_info.assert_any_call("Finished cleaning object columns.")
    
    # Check if the object columns have been cleaned correctly
    assert cleaned_df['Name'].tolist() == ['john', 'jane', 'doe']
    assert cleaned_df['City'].tolist() == ['new york', 'london', 'paris']
    assert cleaned_df['Age'].tolist() == [25, 30, 22]  # Age should be unchanged

import pytest
import pandas as pd
from your_module_name import delete_columns
from unittest.mock import patch

# Define the parameterized test using pytest.mark.parametrize
@pytest.mark.parametrize(
    "initial_columns, columns_to_drop, expected_columns",
    [
        (['A', 'B', 'C'], 'A', ['B', 'C']),   # Test single column drop
        (['A', 'B', 'C'], ['A', 'B'], ['C']), # Test dropping multiple columns
        (['A', 'B', 'C'], 'D', ['A', 'B', 'C']),  # Test trying to drop non-existent column
        (['A', 'B', 'C'], ['D', 'E'], ['A', 'B', 'C']),  # Test trying to drop multiple non-existent columns
        (['A', 'B', 'C'], ['A', 'D'], ['B', 'C']),  # Test drop one valid and one non-existent column
    ]
)
@patch('your_module_name.logging.info')  # Mock the logging.info function
def test_delete_columns(mock_logging_info, initial_columns, columns_to_drop, expected_columns):
    # Create a DataFrame with the initial columns
    df = pd.DataFrame(columns=initial_columns)
    
    # Call the delete_columns function
    delete_columns(df, columns_to_drop)
    
    # Verify that the remaining columns in the DataFrame match the expected_columns
    assert list(df.columns) == expected_columns

    # Check that logging.info was called with the correct messages
    mock_logging_info.assert_any_call(f"Attempting to drop columns: {columns_to_drop}")
    mock_logging_info.assert_any_call(f"Successfully dropped columns: {columns_to_drop}")


