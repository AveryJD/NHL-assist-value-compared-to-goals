# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


def filter_players_by_toi(data_years: list, min_toi: int) -> pd.DataFrame:
    """
    Reads player data from CSV files for each season and filters out players who do not meet
    the minimum TOI (time on ice) requirement across all specified seasons.

    :param data_years: a list of strings, where each string is a season corresponding to a CSV filename in the 'data/' directory
    :param min_toi: an int representing the minimum TOI a player must have in each season to be included
    :return: List of DataFrames (one per season), each containing only the players who meet the TOI threshold in all specified seasons
    """

    dataframes = []
    for year in data_years:
        dataframes.append(pd.read_csv(f'data/{year}.csv'))
                          
    eligible_players = set(dataframes[0].loc[dataframes[0]['TOI'] >= min_toi, 'Player'])
    for df in dataframes[1:]:
        eligible_players &= set(df.loc[df['TOI'] >= min_toi, 'Player'])
    filtered_dataframes = [df[df['Player'].isin(eligible_players)].reset_index(drop=True) for df in dataframes]
    
    return filtered_dataframes


def calc_goals_created_per_sixty(df, p_weight, s_weight):
    """
    Calculates the Goals Created per 60 minutes (GC/60) for each player in the provided DataFrame
    using weighted values for primary and secondary assists.

    :param df: a DataFrame containing player stats
    :param p_weight: a float representing the weight to assign to primary assists.
    :param s_weight: a float representing the weight to assign to secondary assists.
    :return: a DataFrame with two columns: 'Player' and 'GC/60'.
    """

    df = df.copy()
    df['Goals Created'] = df['Goals'] + (p_weight * df['First Assists']) + (s_weight * df['Second Assists'])
    df['GC/60'] = df['Goals Created'] / df['TOI'] * 60
    
    return df[['Player', 'GC/60']]


# Initialize values
data_paths = ['2018-2019', '2019-2020', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
min_tois = [0, 100, 200, 300, 400, 500, 600, 700, 800]
primary_weights = np.arange(0, 1.01, 0.02)
secondary_weights = np.arange(0, 1.01, 0.02)

# Lists to keep track of key results
best_p_weights = []
best_s_weights = []
best_r_squareds = []
zero_r_squareds = []
half_r_squareds = []
one_r_squareds = []


# Run analysis for each minimum TOI value
for min_toi in min_tois:

    print(f'\nProcessing TOI ≥ {min_toi}:')

    # Filter players who meet the TOI threshold in every season
    df_2018, df_2019, df_2021, df_2022, df_2023, df_2024 = filter_players_by_toi(data_paths, min_toi)

    r_squared_matrix = np.zeros((len(primary_weights), len(secondary_weights)))

    # Test every possible primary and secondary weight combinations
    for i, p_weight in enumerate(primary_weights):
        for j, s_weight in enumerate(secondary_weights):

            # Calculate player goals created for each year
            gc_2018 = calc_goals_created_per_sixty(df_2018, p_weight, s_weight).rename(columns={'GC/60': 'GC/60 2018'})
            gc_2019 = calc_goals_created_per_sixty(df_2019, p_weight, s_weight).rename(columns={'GC/60': 'GC/60 2019'})
            gc_2021 = calc_goals_created_per_sixty(df_2021, p_weight, s_weight).rename(columns={'GC/60': 'GC/60 2021'})
            gc_2022 = calc_goals_created_per_sixty(df_2022, p_weight, s_weight).rename(columns={'GC/60': 'GC/60 2022'})
            gc_2023 = calc_goals_created_per_sixty(df_2023, p_weight, s_weight).rename(columns={'GC/60': 'GC/60 2023'})
            gc_2024 = calc_goals_created_per_sixty(df_2024, p_weight, s_weight).rename(columns={'GC/60': 'GC/60 2024'})

            # Combine GC DataFrame for "predictor" seasons
            merged = gc_2018.merge(gc_2019, on='Player') \
                            .merge(gc_2021, on='Player') \
                            .merge(gc_2022, on='Player') \
                            .merge(gc_2023, on='Player') \
                            .merge(gc_2024, on='Player')

            # Compute weighted average of past GC/60 values to use as predictor
            merged['Prediction GC/60'] = (
                merged['GC/60 2018'] * 0.1 +
                merged['GC/60 2019'] * 0.1 +
                merged['GC/60 2021'] * 0.3 +
                merged['GC/60 2022'] * 0.5 
            )

            # Calculate average R² across two years to assess prediction quality
            X = merged[['Prediction GC/60']]
            y_2023 = merged['GC/60 2023']
            y_2024 = merged['GC/60 2024']
            r_squared = (LinearRegression().fit(X, y_2023).score(X, y_2023) +
                  LinearRegression().fit(X, y_2024).score(X, y_2024)) / 2

            r_squared_matrix[i, j] = r_squared


    # Find R² when both weights are 0.0 for comparison
    p_zero_index = list(primary_weights).index(0.0)
    s_zero_index = list(secondary_weights).index(0.0)
    zero_r_squared = r_squared_matrix[p_zero_index, s_zero_index]

    # Find R² when both weights are 0.5 for comparison
    p_half_index = list(primary_weights).index(0.5)
    s_half_index = list(secondary_weights).index(0.5)
    half_r_squared = r_squared_matrix[p_half_index, s_half_index]

    # Find R² when both weights are 1.0 for comparison
    p_one_index = list(primary_weights).index(1.0)
    s_one_index = list(secondary_weights).index(1.0)
    one_r_squared = r_squared_matrix[p_one_index, s_one_index]

    # Find the best combination of weights (highest R²)
    best_index = np.unravel_index(np.argmax(r_squared_matrix), r_squared_matrix.shape)
    best_p = primary_weights[best_index[0]]
    best_s = secondary_weights[best_index[1]]
    best_r_squared = r_squared_matrix[best_index]

    print(f"Best primary assist weight: {best_p:.2f}\nBest secondary assist weight: {best_s:.2f}\nR² = {best_r_squared:.4f}")

    # Create predicted vs actual plot data
    merged['Actual GC/60'] = (merged['GC/60 2023'] + merged['GC/60 2024']) / 2
    plot_df = merged[['Player', 'Prediction GC/60', 'Actual GC/60']].copy()
    tot_players = len(plot_df)

    # Make scatter plot of predicted goals created vs actual goals created
    plt.figure(figsize=(8, 8))
    plt.plot([0, 3.5], [0, 3.5], linestyle='--', color='red')
    plt.scatter(plot_df['Actual GC/60'], plot_df['Prediction GC/60'], alpha=0.7)
    
    plt.xlabel('Actual GC/60')
    plt.xlim(0, 3.5)
    plt.ylabel('Predicted GC/60')
    plt.ylim(0, 3.5)

    plt.title(f'Predicted vs Actual GC/60\nTOI ≥ {min_toi} | Total Players = {tot_players}')
    plt.text(0.05, 3.35, f'R²={best_r_squared:.3f}', fontsize = 20)
    plt.tight_layout()
    plt.savefig(f'plots/scatterplot_{min_toi}_toi.png')
    plt.close()

    # Make heatmap of R² values by weight combo
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        r_squared_matrix.T,
        xticklabels=[f"{p:.2f}" for p in primary_weights],
        yticklabels=[f"{s:.2f}" for s in secondary_weights],
        cmap='RdYlGn',
        annot=False,
    )

    plt.xlabel('Primary Assist Weight')
    plt.ylabel('Secondary Assist Weight')
    plt.gca().invert_yaxis()

    plt.title(f'R² Heatmap (TOI ≥ {min_toi})')
    plt.tight_layout()
    plt.savefig(f'plots/heatmap_{min_toi}_toi.png')
    plt.close()


    # Add values to lists for comparison plot
    best_p_weights.append(best_p)
    best_s_weights.append(best_s)
    best_r_squareds.append(best_r_squared)
    zero_r_squareds.append(zero_r_squared)
    half_r_squareds.append(half_r_squared)
    one_r_squareds.append(one_r_squared)


# Make line graph of R² values with different assist weightings
plt.figure(figsize=(10, 8))
plt.plot(min_tois, best_r_squareds, label='R² When Best Weights are found', color='green', marker='o')
plt.plot(min_tois, zero_r_squareds, label='R² When Both Weights are 0.0', color='red', marker='o')
plt.plot(min_tois, half_r_squareds, label='R² When Both Weights are 0.5', color='orange', marker='o')
plt.plot(min_tois, one_r_squareds, label='R² When Both Weights are 1.0', color='yellow', marker='o')

plt.xlabel('Minimum TOI Requirement')
plt.xlim(-10, 810)
plt.xticks(min_tois)
plt.ylabel('R²')
plt.ylim(0, 1.0)
plt.yticks(np.arange(0.1, 1.1, 0.1))

plt.title('R²s vs Minimum TOI Requirements')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/linegraph_r_squared.png')
plt.close()



# Make line graph of best assist weightings
plt.figure(figsize=(10, 8))
plt.plot(min_tois, best_p_weights, label='Best Primary Assist Weights', color='blue', marker='o')
plt.plot(min_tois, best_s_weights, label='Best Secondary Assist Weights', color='deepskyblue', marker='o')

plt.xlabel('Minimum TOI Requirement')
plt.xlim(-10, 810)
plt.xticks(min_tois)
plt.ylabel('Assist Weights')
plt.ylim(0, 1.0)
plt.yticks(np.arange(0.1, 1.1, 0.1))

plt.title('Best Assist Weights vs Minimum TOI Requirements')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/linegraph_assist_weights.png')
plt.close()
