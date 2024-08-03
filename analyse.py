import pandas as pd

def analyze_data(data):
    # Calculate statistics
    mean_reward = round(data['reward'].mean(), 2)       # Mittelwert
    median_reward = data['reward'].median()             # Median / Q2
    min_reward = data['reward'].min()                   # Min
    max_reward = data['reward'].max()                   # Max
    q1_reward = data['reward'].quantile(0.25)           # Q1
    q3_reward = data['reward'].quantile(0.75)           # Q3
    iqr_reward = q3_reward - q1_reward                  # IQR

    success_rate = round((data['reward'] == 100).sum() / data.shape[0] * 100, 2)
    three_heroes = round((data['reward'] == 60).sum() / data.shape[0] * 100, 2)
    two_heroes = round((data['reward'] == 40).sum() / data.shape[0] * 100, 2)
    one_heroes = round((data['reward'] == 20).sum() / data.shape[0] * 100, 2)
    failure_rate = round((data['reward'] == -100).sum() / data.shape[0] * 100, 2)
    run_away_rate = round((data['reward'] == -50).sum() / data.shape[0] * 100, 2)

    # Print the results
    print("############### RESULTS ###############")
    print(f"Mean: {mean_reward}")
    print(f"Q1: {q1_reward}")
    print(f"Median: {median_reward}")
    print(f"Q3: {q3_reward}")
    print(f"Min: {min_reward}")
    print(f"Max: {max_reward}")
    print(f"IQR (Interquartile Range): {iqr_reward}\n")

    print(f"All heroes survive: {success_rate} %")
    print(f"3 heroes survive: {three_heroes} %")
    print(f"2 heroes survive: {two_heroes} %")
    print(f"1 heroes survive: {one_heroes} %")
    print(f"Heroes die: {failure_rate} %")
    print(f"Heros run away: {run_away_rate} %")

def analyze_action_spread(data):
    # Function to calculate counts and frequencies
    def calculate_counts_and_frequencies(column):
        counts = column.value_counts().sort_index()
        frequencies = counts / len(column) * 100
        return counts, frequencies

    # Dictionary to store results
    results = {}

    # Analyze each class (column)
    for column in data.columns:
        counts, frequencies = calculate_counts_and_frequencies(data[column])
        results[column] = {
            'Counts': counts.to_dict(),
            'Frequencies (%)': frequencies.to_dict()
        }

    # Print results
    for class_name, result in results.items():
        print(f"\n{class_name}:")
        print("Counts:")
        for action, count in result['Counts'].items():
            print(f"  Action {action}: {count}")
        print("Frequencies (%):")
        for action, freq in result['Frequencies (%)'].items():
            print(f"  Action {action}: {freq:.2f}%")