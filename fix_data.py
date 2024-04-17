import pandas as pd

def load_data (file_name):
    try:
        data = pd.read_csv(f'./src/data/{file_name}.csv')
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {e}")
    
def divide_data_by_token(data):
    grouped_tokens = data.groupby("Id")
    token_dataframes = [group.copy() for _, group in grouped_tokens]
    return token_dataframes

def filter_tokens_by_record_count(token_dataframes, min_records = 289):
    filtered_dataframes = []
    for df in token_dataframes:
        if len(df) >= min_records:
            filtered_dataframes.append(df)
    return filtered_dataframes

def save_to_csv(data, file_name):
    # Save filtered data to CSV
    pd.concat(data).to_csv(file_name, index=False)

def main():
    try:
        # Load the CSV file
        data_file_name = 'training_ai_data01-02_03_24'
        data = load_data(data_file_name)

        # Divide DataFrame into multiple DataFrames for each token
        token_dataframes = divide_data_by_token(data)

        # Filter tokens by record count
        filtered_dataframes = filter_tokens_by_record_count(token_dataframes, min_records=289)

        # Save filtered data to new CSV file
        save_to_csv(filtered_dataframes, 'new_data.csv')

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
