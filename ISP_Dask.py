import dask.dataframe as dd
import re
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd

# Load the data with explicit dtype specification
file_path = '~/Desktop/ISP Project/Rookie_ADP.csv'
data = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Function to clean and swap the player names
def clean_player_name(player):
    # Remove everything after and including the first bracket
    player = re.split(r'\(|\[', player)[0].strip()
    # Remove periods
    player = player.replace('.', '')
    # Swap the names
    if ',' in player:
        last_name, first_name = player.split(', ')
        player = f"{first_name} {last_name}"
    return player

# Apply the function to the Player column
data['Player'] = data['Player'].map(clean_player_name, meta=('Player', 'object'))

# Save the cleaned data to a new CSV file
cleaned_file_path = '~/Desktop/ISP Project/Cleaned_Rookie_ADP_New.csv'
data.to_csv(cleaned_file_path, index=False, single_file=True)

print("Cleaned data saved to:", cleaned_file_path)

# Define the file paths
cleaned_file_path = os.path.expanduser('~/Desktop/ISP Project/Cleaned_Rookie_ADP_New.csv')
stats_file_path = os.path.expanduser('~/Desktop/ISP Project/Stats.csv')

# Load the cleaned data with explicit dtype specification
cleaned_data = dd.read_csv(cleaned_file_path, dtype={'Age': 'float64'})

# Load the stats data with explicit dtype specification
stats_data = dd.read_csv(stats_file_path, dtype={'Age': 'float64'})

# Clean player names in stats data
stats_data['Player'] = stats_data['Player'].astype(str).map(clean_player_name, meta=('Player', 'object'))

# Filter players who played more than 10 games and play the positions WR, TE, RB, and QB
positions_of_interest = ['WR', 'TE', 'RB', 'QB']
filtered_stats = stats_data[(stats_data['Games_Played'] > 10) & (stats_data['Pos'].isin(positions_of_interest))]

# Compute the values of 'Player' column in cleaned_data to a list
cleaned_players = cleaned_data['Player'].compute().tolist()

# Find players not listed in the cleaned data
missing_players = filtered_stats[~filtered_stats['Player'].isin(cleaned_players)]['Player'].compute()

# Define the output file path
output_file_path = os.path.expanduser('~/Desktop/ISP Project/Missing_Players.csv')

# Save the missing players list to a new CSV file
missing_players.to_csv(output_file_path, index=False, header=['Player'])

print("Missing players saved to:", output_file_path)

# Load the cleaned rookie data with explicit dtype specification
cleaned_data = dd.read_csv(cleaned_file_path, dtype={'Age': 'float64'})

# Load the stats data with explicit dtype specification
stats_data = dd.read_csv(stats_file_path, dtype={'Age': 'float64'})

# Clean player names in stats data
stats_data['Player'] = stats_data['Player'].astype(str).map(clean_player_name, meta=('Player', 'object'))

# Merge the stats data into the cleaned rookie data
merged_data = dd.merge(cleaned_data, stats_data, on='Player', how='left')

# Save the merged data to a new CSV file
merged_file_path = os.path.expanduser('~/Desktop/ISP Project/Merged_Rookie_Stats.csv')
merged_data.to_csv(merged_file_path, index=False, single_file=True)

print("Merged data saved to:", merged_file_path)

# Load the CSV file
file_path = '~/Desktop/ISP Project/Merged_Rookie_Stats.csv'
df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Define the scoring criteria
scoring_criteria = {
    'Pass_yards': 0.1,
    'Pass_TD': 6,
    'Pass_Int': -2,
    'Rush_yards': 0.1,
    'Rush_TD': 6,
    'Rec_rec': 0.5,
    'Rec_yards': 0.1,
    'Rec_TD': 6
}

# Calculate the Total Fantasy Points
df['Total Fantasy Points'] = (
    df['Pass_yards'] * scoring_criteria['Pass_yards'] +
    df['Pass_TD'] * scoring_criteria['Pass_TD'] +
    df['Pass_Int'] * scoring_criteria['Pass_Int'] +
    df['Rush_yards'] * scoring_criteria['Rush_yards'] +
    df['Rush_TD'] * scoring_criteria['Rush_TD'] +
    df['Rec_rec'] * scoring_criteria['Rec_rec'] +
    df['Rec_yards'] * scoring_criteria['Rec_yards'] +
    df['Rec_TD'] * scoring_criteria['Rec_TD']
)

# Calculate the Fantasy Points Per Game
df['Fantasy Points Per Game'] = df['Total Fantasy Points'] / df['Games_Played']

# Save the updated dataframe to a new CSV file
updated_file_path = '~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv'
df.to_csv(updated_file_path, index=False, single_file=True)

print(f"Updated file saved to {updated_file_path}")

# Load the updated CSV file
file_path = os.path.expanduser('~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv')
df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Extract the draft round from the Draft_Rank column
df['Draft_Round'] = df['Draft_Rank'].apply(lambda x: int(str(x).split('.')[0]), meta=('Draft_Round', 'int'))

# Group by Draft_Round and calculate the average Fantasy Points Per Game
rounds_avg_fppg = df.groupby('Draft_Round')['Fantasy Points Per Game'].mean().compute()

# Create Charts folder if it doesn't exist
charts_folder = os.path.expanduser('~/Desktop/ISP Project/Charts')
if not os.path.exists(charts_folder):
    os.makedirs(charts_folder)

# Plot average Fantasy Points Per Game by Draft Round
plt.figure(figsize=(10, 6))
rounds_avg_fppg.plot(kind='bar')
plt.xlabel('Draft Round')
plt.ylabel('Average Fantasy Points Per Game')
plt.title('Average Fantasy Points Per Game by Draft Round')
plt.xticks(rotation=0)

# Save the chart as an image
chart_path = os.path.join(charts_folder, 'Average_Fantasy_Points_Per_Game_by_Draft_Round.png')
plt.savefig(chart_path)
plt.close()

# Save the dataframe to a new CSV file for further use
avg_fppg_file_path = os.path.expanduser('~/Desktop/ISP Project/Average_Fantasy_Points_Per_Game_by_Draft_Round.csv')
rounds_avg_fppg.to_csv(avg_fppg_file_path)

print(f"Chart saved to {chart_path}")
print(f"Updated file saved to {avg_fppg_file_path}")

# Load the updated CSV file
file_path = os.path.expanduser('~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv')
df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Display unique values in the 'Pos' column to identify any NaNs or unexpected values
unique_positions = df['Pos'].unique().compute()
print("Unique values in 'Pos' column:")
print(unique_positions)

# Define a color map for positions, including a default color for NaN or unexpected values
position_colors = {
    'QB': 'blue',
    'WR': 'green',
    'RB': 'red',
    'TE': 'purple',
}

# Fill NaN values in 'Pos' with a placeholder, e.g., 'Unknown'
df['Pos'] = df['Pos'].fillna('Unknown')

# Map the positions to colors, using a default color for 'Unknown'
df['Color'] = df['Pos'].map(lambda x: position_colors.get(x, 'grey'), meta=('Color', 'object'))

# Compute the dataframe
df = df.compute()

# Create scatter plots for each position, excluding NaNs
positions = df['Pos'].unique()
charts_folder = os.path.expanduser('~/Desktop/ISP Project/Charts')

for pos in positions:
    if pos == 'Unknown':
        continue  # Skip NaN/Unknown positions
    plt.figure(figsize=(10, 6))
    subset = df[df['Pos'] == pos]
    plt.scatter(subset['Draft_Rank'], subset['Total Fantasy Points'], alpha=0.5, color=position_colors.get(pos, 'grey'), label=pos)
    plt.xlabel('Draft Rank')
    plt.ylabel('Total Fantasy Points')
    plt.title(f'Total Fantasy Points vs. Draft Rank ({pos})')
    plt.legend(title='Position')
    plt.grid(True)
    plt.xticks(ticks=range(int(df['Draft_Rank'].min()), int(df['Draft_Rank'].max()) + 1))  # Set x-axis to whole numbers

    # Save the chart as an image
    chart_path = os.path.join(charts_folder, f'Total_Fantasy_Points_vs_Draft_Rank_{pos}.png')
    plt.savefig(chart_path)
    plt.close()

# Calculate the correlation between Draft Rank and Total Fantasy Points
correlation = df['Draft_Rank'].corr(df['Total Fantasy Points'])
print(f"Correlation between Draft Rank and Total Fantasy Points: {correlation}")

# Save the dataframe to a new CSV file for further use
comparison_file_path = os.path.expanduser('~/Desktop/ISP Project/Total_Fantasy_Points_vs_Draft_Rank.csv')
df[['Draft_Rank', 'Total Fantasy Points']].to_csv(comparison_file_path, index=False)

print(f"Updated file saved to {comparison_file_path}")

# Load the updated CSV file
file_path = os.path.expanduser('~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv')
df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Extract the draft round from the Draft_Rank column
df['Draft_Round'] = df['Draft_Rank'].apply(lambda x: int(str(x).split('.')[0]), meta=('Draft_Round', 'int'))

# Calculate the number of Pro Bowl and All-Pro players for each draft round
pro_bowl_counts = df[df['Pro_Bowl'] == 1].groupby('Draft_Round').size().compute()
all_pro_counts = df[df['All_Pro'] == 1].groupby('Draft_Round').size().compute()

# Combine the counts into a single DataFrame
comparison_df = pd.DataFrame({
    'Pro_Bowl': pro_bowl_counts,
    'All_Pro': all_pro_counts
}).fillna(0)

# Plot the comparison
plt.figure(figsize=(12, 8))
comparison_df.plot(kind='bar')
plt.xlabel('Draft Round')
plt.ylabel('Number of Players')
plt.title('Number of Pro Bowl and All-Pro Players by Draft Round')
plt.legend(title='Achievement', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Create the path to save the chart
charts_folder = os.path.expanduser('~/Desktop/ISP Project/Charts')
chart_path = os.path.join(charts_folder, 'Pro_Bowl_and_All_Pro_by_Draft_Round.png')
plt.savefig(chart_path)
plt.close()

# Save the comparison DataFrame to a new CSV file for further use
comparison_file_path = os.path.expanduser('~/Desktop/ISP Project/Pro_Bowl_and_All_Pro_by_Draft_Round.csv')
comparison_df.to_csv(comparison_file_path)

print(f"Chart saved to {chart_path}")
print(f"Updated file saved to {comparison_file_path}")

# Load the updated CSV file
file_path = os.path.expanduser('~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv')
df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Group by College/Univ and calculate the mean draft position, total points scored, and number of players drafted
college_stats = df.groupby('College/Univ').agg({
    'Draft_Rank': 'mean',
    'Total Fantasy Points': 'sum',
    'Player': 'count'  # Assuming 'Player' column contains player names or IDs
}).compute().reset_index()

# Rename columns for better understanding
college_stats.columns = ['College/Univ', 'Average Draft Position', 'Total Fantasy Points', 'Number of Players Drafted']

# Sort the colleges by total fantasy points scored and average draft position
top_colleges_by_points = college_stats.sort_values(by='Total Fantasy Points', ascending=False).head(10)
top_colleges_by_draft = college_stats.sort_values(by='Average Draft Position').head(10)

# Display the dataframes
print("Top 10 Colleges/Univ by Total Fantasy Points:")
print(top_colleges_by_points)

print("\nTop 10 Colleges/Univ by Average Draft Position:")
print(top_colleges_by_draft)

# Create the path to save the charts
charts_folder = os.path.expanduser('~/Desktop/ISP Project/Charts')

# Plot the results

# Top 10 Colleges/Univ by Total Fantasy Points
fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx()
bar_width = 0.35

top_colleges_by_points.plot(kind='bar', x='College/Univ', y='Total Fantasy Points', ax=ax1, position=0, width=bar_width, color='skyblue', legend=False)
top_colleges_by_points.plot(kind='bar', x='College/Univ', y='Number of Players Drafted', ax=ax2, position=1, width=bar_width, color='orange', legend=False)

ax1.set_xlabel('College/Univ')
ax1.set_ylabel('Total Fantasy Points')
ax2.set_ylabel('Number of Players Drafted')
ax1.set_title('Top 10 Colleges/Univ by Total Fantasy Points and Number of Players Drafted')
ax1.set_xticks(range(len(top_colleges_by_points)))
ax1.set_xticklabels(top_colleges_by_points['College/Univ'], rotation=45, ha='right')

# Add legends
ax1.legend(['Total Fantasy Points'], loc='upper left')
ax2.legend(['Number of Players Drafted'], loc='upper right')

# Save the chart as an image
chart_path = os.path.join(charts_folder, 'Top_10_Colleges_by_Total_Fantasy_Points.png')
fig.savefig(chart_path)
plt.close(fig)

# Top 10 Colleges/Univ by Average Draft Position
fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx()

top_colleges_by_draft.plot(kind='bar', x='College/Univ', y='Average Draft Position', ax=ax1, position=0, width=bar_width, color='orange', legend=False)
top_colleges_by_draft.plot(kind='bar', x='College/Univ', y='Number of Players Drafted', ax=ax2, position=1, width=bar_width, color='skyblue', legend=False)

ax1.set_xlabel('College/Univ')
ax1.set_ylabel('Average Draft Position')
ax2.set_ylabel('Number of Players Drafted')
ax1.set_title('Top 10 Colleges/Univ by Average Draft Position and Number of Players Drafted')
ax1.set_xticks(range(len(top_colleges_by_draft)))
ax1.set_xticklabels(top_colleges_by_draft['College/Univ'], rotation=45, ha='right')

# Add legends
ax1.legend(['Average Draft Position'], loc='upper left')
ax2.legend(['Number of Players Drafted'], loc='upper right')

# Save the chart as an image
chart_path = os.path.join(charts_folder, 'Top_10_Colleges_by_Average_Draft_Position.png')
fig.savefig(chart_path)
plt.close(fig)

# Save the dataframes to new CSV files for further use
top_colleges_by_points_file_path = os.path.expanduser('~/Desktop/ISP Project/Top_10_Colleges_by_Total_Fantasy_Points.csv')
top_colleges_by_draft_file_path = os.path.expanduser('~/Desktop/ISP Project/Top_10_Colleges_by_Average_Draft_Position.csv')

top_colleges_by_points.to_csv(top_colleges_by_points_file_path, index=False)
top_colleges_by_draft.to_csv(top_colleges_by_draft_file_path, index=False)

print(f"Files saved to {top_colleges_by_points_file_path} and {top_colleges_by_draft_file_path}")

# Load the updated CSV file
file_path = os.path.expanduser('~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv')
df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Extract the draft round from the Draft_Rank column
df['Draft_Round'] = df['Draft_Rank'].apply(lambda x: int(str(x).split('.')[0]), meta=('Draft_Round', 'int'))

# Group by position and draft round to calculate total and average fantasy points
position_draft_stats = df.groupby(['Pos', 'Draft_Round']).agg({
    'Total Fantasy Points': ['sum', 'mean'],
    'Player': 'count'  # Assuming 'Player' column contains player names or IDs
}).compute().reset_index()

# Flatten the multi-index columns
position_draft_stats.columns = ['Position', 'Draft_Round', 'Total Fantasy Points', 'Average Fantasy Points', 'Number of Players Drafted']

# Display the dataframe
print("Total and Average Fantasy Points by Position and Draft Round:")
print(position_draft_stats)

# Create the path to save the charts
charts_folder = os.path.expanduser('~/Desktop/ISP Project/Charts')

# Plot the results

# Total Fantasy Points by Position and Draft Round
fig, ax = plt.subplots(figsize=(14, 7))
position_draft_stats.pivot(index='Draft_Round', columns='Position', values='Total Fantasy Points').plot(kind='bar', stacked=True, ax=ax)
plt.xlabel('Draft Round')
plt.ylabel('Total Fantasy Points')
plt.title('Total Fantasy Points by Position and Draft Round')
plt.xticks(rotation=0)
plt.legend(title='Position', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the chart as an image
chart_path = os.path.join(charts_folder, 'Total_Fantasy_Points_by_Position_and_Draft_Round.png')
fig.savefig(chart_path)
plt.close(fig)

# Average Fantasy Points by Position and Draft Round
fig, ax = plt.subplots(figsize=(14, 7))
position_draft_stats.pivot(index='Draft_Round', columns='Position', values='Average Fantasy Points').plot(kind='bar', stacked=True, ax=ax)
plt.xlabel('Draft Round')
plt.ylabel('Average Fantasy Points')
plt.title('Average Fantasy Points by Position and Draft Round')
plt.xticks(rotation=0)
plt.legend(title='Position', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the chart as an image
chart_path = os.path.join(charts_folder, 'Average_Fantasy_Points_by_Position_and_Draft_Round.png')
fig.savefig(chart_path)
plt.close(fig)

# Save the dataframe to a new CSV file for further use
position_draft_file_path = os.path.expanduser('~/Desktop/ISP Project/Total_and_Average_Fantasy_Points_by_Position_and_Draft_Round.csv')
position_draft_stats.to_csv(position_draft_file_path, index=False)

print(f"File saved to {position_draft_file_path}")

# Load the updated CSV file
file_path = '~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv'
df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Extract the draft round from the Draft_Rank column
df['Draft_Round'] = df['Draft_Rank'].apply(lambda x: int(str(x).split('.')[0]), meta=('Draft_Round', 'int'))

# Calculate the average fantasy points for each position by draft round
position_draft_stats = df.groupby(['Pos', 'Draft_Round']).agg({
    'Total Fantasy Points': 'mean',
    'Player': 'count'
}).compute().reset_index()

# Flatten the multi-index columns
position_draft_stats.columns = ['Position', 'Draft_Round', 'Average Fantasy Points', 'Number of Players Drafted']

# Calculate the average fantasy points for each position by draft round
avg_fantasy_points = position_draft_stats.pivot(index='Draft_Round', columns='Position', values='Average Fantasy Points')

# Display the pivot table
print("\nAverage Fantasy Points by Position and Draft Round (Pivot Table):")
print(avg_fantasy_points)

# Define the positions needed
positions_needed = ['QB', 'WR', 'WR', 'RB', 'RB', 'TE', 'Flex', 'Flex']

# Determine the optimal draft strategy
draft_strategy = []
available_positions = positions_needed.copy()

for round_num in range(1, 6):  # Assuming we have 5 rounds to pick
    best_position = None
    max_points = 0
    for position in available_positions:
        if position in avg_fantasy_points.columns and round_num in avg_fantasy_points.index:
            points = avg_fantasy_points.loc[round_num, position]
            if points > max_points:
                max_points = points
                best_position = position
    if best_position:
        draft_strategy.append((round_num, best_position, max_points))
        available_positions.remove(best_position)

# Fill remaining positions with best available flex options
for round_num in range(len(draft_strategy) + 1, 6):
    best_position = None
    max_points = 0
    for position in ['RB', 'WR', 'TE']:  # Flex options
        if position in avg_fantasy_points.columns and round_num in avg_fantasy_points.index:
            points = avg_fantasy_points.loc[round_num, position]
            if points > max_points:
                max_points = points
                best_position = position
    if best_position:
        draft_strategy.append((round_num, 'Flex', max_points))
        available_positions.remove(best_position)

# Display the draft strategy
draft_strategy_df = pd.DataFrame(draft_strategy, columns=['Round', 'Position', 'Average Fantasy Points'])
print("\nOptimal Draft Strategy:")
print(draft_strategy_df)

# Save the draft strategy to a new CSV file for further use
draft_strategy_file_path = '~/Desktop/ISP Project/Optimal_Draft_Strategy.csv'
draft_strategy_df.to_csv(draft_strategy_file_path, index=False)

print(f"Draft strategy file saved to {draft_strategy_file_path}")

# Load the updated CSV file
file_path = '~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv'
df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Extract the draft pick position from the Draft_Rank column
df['Draft_Pick'] = df['Draft_Rank'].apply(lambda x: float(x), meta=('Draft_Pick', 'float'))

# Group by position and draft pick to calculate average fantasy points
position_pick_stats = df.groupby(['Pos', 'Draft_Pick']).agg({
    'Total Fantasy Points': 'mean',
    'Player': 'count'
}).compute().reset_index()

# Flatten the multi-index columns
position_pick_stats.columns = ['Position', 'Draft_Pick', 'Average Fantasy Points', 'Number of Players Drafted']

# Filter for first round picks (1.01 to 1.12)
first_round_picks = position_pick_stats[position_pick_stats['Draft_Pick'].between(1.01, 1.12)]

# Pivot table for average fantasy points by position and draft pick
avg_fantasy_points_first_round = first_round_picks.pivot(index='Draft_Pick', columns='Position', values='Average Fantasy Points')

# Display the pivot table
print("\nAverage Fantasy Points by Position and Draft Pick (1.01 to 1.12):")
print(avg_fantasy_points_first_round)

# Define the positions needed
positions_needed = ['QB', 'WR', 'WR', 'RB', 'RB', 'TE', 'Flex', 'Flex']

# Determine the optimal draft strategy for each pick in the first round
draft_strategy_first_round = []
available_positions = positions_needed.copy()

for pick in sorted(avg_fantasy_points_first_round.index):
    best_position = None
    max_points = 0
    for position in available_positions:
        if position in avg_fantasy_points_first_round.columns:
            points = avg_fantasy_points_first_round.loc[pick, position]
            if points > max_points:
                max_points = points
                best_position = position
    if best_position:
        draft_strategy_first_round.append((pick, best_position, max_points))
        available_positions.remove(best_position)

# Fill remaining positions with best available WR, RB, or TE (for Flex positions)
for pick in sorted(avg_fantasy_points_first_round.index):
    if len(draft_strategy_first_round) >= 12:
        break
    best_position = None
    max_points = 0
    for position in ['RB', 'WR', 'TE']:  # Flex options
        if position in avg_fantasy_points_first_round.columns:
            points = avg_fantasy_points_first_round.loc[pick, position]
            if points > max_points:
                max_points = points
                best_position = position
    if best_position and best_position in available_positions:
        draft_strategy_first_round.append((pick, best_position, max_points))
        available_positions.remove(best_position)

# Display the draft strategy for the first round
draft_strategy_first_round_df = pd.DataFrame(draft_strategy_first_round, columns=['Draft Pick', 'Position', 'Average Fantasy Points'])
print("\nOptimal Draft Strategy for First Round (1.01 to 1.12):")
print(draft_strategy_first_round_df)

# Save the draft strategy to a new CSV file for further use
draft_strategy_first_round_file_path = '~/Desktop/ISP Project/Optimal_Draft_Strategy_First_Round.csv'
draft_strategy_first_round_df.to_csv(draft_strategy_first_round_file_path, index=False)

print(f"Draft strategy file saved to {draft_strategy_first_round_file_path}")

# Load the updated CSV file
file_path = '~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv'
df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Extract the draft pick position from the Draft_Rank column
df['Draft_Pick'] = df['Draft_Rank'].apply(lambda x: float(x), meta=('Draft_Pick', 'float'))

# Group by position and draft pick to calculate average fantasy points
position_pick_stats = df.groupby(['Pos', 'Draft_Pick']).agg({
    'Total Fantasy Points': 'mean',
    'Player': 'count'
}).compute().reset_index()

# Flatten the multi-index columns
position_pick_stats.columns = ['Position', 'Draft_Pick', 'Average Fantasy Points', 'Number of Players Drafted']

# Pivot table for average fantasy points by position and draft pick
avg_fantasy_points = position_pick_stats.pivot(index='Draft_Pick', columns='Position', values='Average Fantasy Points')

# Display the pivot table
print("\nAverage Fantasy Points by Position and Draft Pick:")
print(avg_fantasy_points)

# Calculate the historical value for each draft pick by averaging the values across all positions
pick_values = avg_fantasy_points.mean(axis=1).reset_index()
pick_values.columns = ['Draft_Pick', 'Historical Value']

# Display the historical values for all picks
print("\nHistorical Values for All Picks:")
print(pick_values)

# Save the historical values to a new CSV file for further use
historical_values_file_path = '~/Desktop/ISP Project/Historical_Values_for_All_Picks.csv'
pick_values.to_csv(historical_values_file_path, index=False)

print(f"Historical values file saved to {historical_values_file_path}")

# Load the updated CSV file
file_path = '~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv'
df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Extract the draft pick position from the Draft_Rank column
df['Draft_Pick'] = df['Draft_Rank'].apply(lambda x: float(x), meta=('Draft_Pick', 'float'))

# Group draft picks into groups of three
def group_picks(pick):
    if 1.01 <= pick <= 1.03:
        return '1.01-1.03'
    elif 1.04 <= pick <= 1.06:
        return '1.04-1.06'
    elif 1.07 <= pick <= 1.09:
        return '1.07-1.09'
    elif 1.10 <= pick <= 1.12:
        return '1.10-1.12'
    elif 2.01 <= pick <= 2.03:
        return '2.01-2.03'
    elif 2.04 <= pick <= 2.06:
        return '2.04-2.06'
    elif 2.07 <= pick <= 2.09:
        return '2.07-2.09'
    elif 2.10 <= pick <= 2.12:
        return '2.10-2.12'
    elif 3.01 <= pick <= 3.03:
        return '3.01-3.03'
    elif 3.04 <= pick <= 3.06:
        return '3.04-3.06'
    elif 3.07 <= pick <= 3.09:
        return '3.07-3.09'
    elif 3.10 <= pick <= 3.12:
        return '3.10-3.12'
    else:
        return '4.01+'

df['Draft_Pick_Group'] = df['Draft_Pick'].apply(group_picks, meta=('Draft_Pick_Group', 'object'))

# Group by position and draft pick group to calculate average fantasy points
position_pick_stats = df.groupby(['Pos', 'Draft_Pick_Group']).agg({
    'Total Fantasy Points': 'mean',
    'Player': 'count'
}).compute().reset_index()

# Flatten the multi-index columns
position_pick_stats.columns = ['Position', 'Draft_Pick_Group', 'Average Fantasy Points', 'Number of Players Drafted']

# Pivot table for average fantasy points by position and draft pick group
avg_fantasy_points_grouped = position_pick_stats.pivot(index='Draft_Pick_Group', columns='Position', values='Average Fantasy Points')

# Display the pivot table
print("\nAverage Fantasy Points by Position and Draft Pick Group:")
print(avg_fantasy_points_grouped)

# Calculate the historical value for each draft pick group by averaging the values across all positions
pick_group_values = avg_fantasy_points_grouped.mean(axis=1).reset_index()
pick_group_values.columns = ['Draft_Pick_Group', 'Historical Value']

# Display the historical values for all draft pick groups
print("\nHistorical Values for All Draft Pick Groups:")
print(pick_group_values)

# Save the historical values to a new CSV file for further use
historical_values_grouped_file_path = '~/Desktop/ISP Project/Historical_Values_for_All_Draft_Pick_Groups.csv'
pick_group_values.to_csv(historical_values_grouped_file_path, index=False)

print(f"Historical values file saved to {historical_values_grouped_file_path}")

# Load the updated CSV file
file_path = os.path.expanduser('~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv')
df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Filter the dataset to include only QB, RB, TE, and WR positions
df = df[df['Pos'].isin(['QB', 'RB', 'TE', 'WR'])]

# Drop rows with missing values in 'Draft_Rank' or 'Total Fantasy Points'
df = df.dropna(subset=['Draft_Rank', 'Total Fantasy Points'])

# Define a color map for positions
position_colors = {
    'QB': 'blue',
    'WR': 'green',
    'RB': 'red',
    'TE': 'purple',
}

# Compute the dataframe
df = df.compute()

# Plot the draft picks vs. total fantasy points
plt.figure(figsize=(14, 7))
for pos in position_colors.keys():
    subset = df[df['Pos'] == pos]
    plt.scatter(subset['Draft_Rank'], subset['Total Fantasy Points'], alpha=0.5, label=pos, color=position_colors[pos])

plt.xlabel('Draft Rank')
plt.ylabel('Total Fantasy Points')
plt.title('Draft Rank vs. Total Fantasy Points by Position')
plt.legend(title='Position')
plt.grid(True)

# Prepare the data for regression
X = df['Draft_Rank'].values.reshape(-1, 1)
y = df['Total Fantasy Points'].values

# Create and fit the regression model
reg = LinearRegression()
reg.fit(X, y)

# Make predictions for the regression line
y_pred = reg.predict(X)

# Plot the regression line
plt.plot(df['Draft_Rank'], y_pred, color='black', linewidth=2, label='Regression Line')

plt.legend()

# Create the path to save the chart
charts_folder = os.path.expanduser('~/Desktop/ISP Project/Charts')
if not os.path.exists(charts_folder):
    os.makedirs(charts_folder)
chart_path = os.path.join(charts_folder, 'Draft_Rank_vs_Total_Fantasy_Points_by_Position.png')
plt.savefig(chart_path)
plt.close()

# Calculate the correlation between Draft Rank and Total Fantasy Points
correlation = df['Draft_Rank'].corr(df['Total Fantasy Points'])
print(f"Correlation between Draft Rank and Total Fantasy Points: {correlation}")

# Display the regression coefficients and R-squared value
intercept = reg.intercept_
slope = reg.coef_[0]
r_squared = r2_score(y, y_pred)

print(f"Regression Coefficients: Intercept = {intercept}, Slope = {slope}")
print(f"R-squared Value: {r_squared}")

# Save the regression results to a new CSV file
regression_results = pd.DataFrame({
    'Draft_Rank': df['Draft_Rank'],
    'Total Fantasy Points': df['Total Fantasy Points'],
    'Predicted Fantasy Points': y_pred
})
regression_results_file_path = os.path.expanduser('~/Desktop/ISP Project/Regression_Results.csv')
regression_results.to_csv(regression_results_file_path, index=False)

print(f"Regression results file saved to {regression_results_file_path}")

# Save the regression coefficients and R-squared value to a new CSV file
regression_summary = pd.DataFrame({
    'Position': list(position_colors.keys()),
    'Intercept': [intercept] * len(position_colors),
    'Slope': [slope] * len(position_colors),
    'R-squared': [r_squared] * len(position_colors)
})
regression_summary_file_path = os.path.expanduser('~/Desktop/ISP Project/Regression_Summary.csv')
regression_summary.to_csv(regression_summary_file_path, index=False)

print(f"Regression summary file saved to {regression_summary_file_path}")

# Load the updated CSV file
file_path = os.path.expanduser('~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv')
df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Filter the dataset to include only QB, RB, TE, and WR positions
df = df[df['Pos'].isin(['QB', 'RB', 'TE', 'WR'])]

# Drop rows with missing values in 'Draft_Rank' or 'Total Fantasy Points'
df = df.dropna(subset=['Draft_Rank', 'Total Fantasy Points'])

# Define a color map for positions
position_colors = {
    'QB': 'blue',
    'WR': 'green',
    'RB': 'red',
    'TE': 'purple',
}

# Compute the dataframe
df = df.compute()

# Initialize lists to store regression summary information
positions = []
intercepts = []
slopes = []
r_squared_values = []

# Generate separate plots for each position
for pos, color in position_colors.items():
    subset = df[df['Pos'] == pos]
    
    # Prepare the data for regression
    X = subset['Draft_Rank'].values.reshape(-1, 1)
    y = subset['Total Fantasy Points'].values

    # Create and fit the regression model
    reg = LinearRegression()
    reg.fit(X, y)

    # Make predictions for the regression line
    y_pred = reg.predict(X)

    # Plot the draft picks vs. total fantasy points
    plt.figure(figsize=(14, 7))
    plt.scatter(subset['Draft_Rank'], subset['Total Fantasy Points'], alpha=0.5, label=pos, color=color)
    plt.plot(subset['Draft_Rank'], y_pred, color='black', linewidth=2, label='Regression Line')

    plt.xlabel('Draft Rank')
    plt.ylabel('Total Fantasy Points')
    plt.title(f'Draft Rank vs. Total Fantasy Points for {pos}')
    plt.legend(title='Position')
    plt.grid(True)

    # Save the chart
    chart_path = os.path.join(charts_folder, f'Draft_Rank_vs_Total_Fantasy_Points_{pos}.png')
    plt.savefig(chart_path)
    plt.close()

    # Calculate regression coefficients and R-squared value
    intercept = reg.intercept_
    slope = reg.coef_[0]
    r_squared = r2_score(y, y_pred)
    
    # Print the results
    print(f"{pos} Regression Coefficients: Intercept = {intercept}, Slope = {slope}")
    print(f"{pos} R-squared Value: {r_squared}")

    # Append the results to the summary lists
    positions.append(pos)
    intercepts.append(intercept)
    slopes.append(slope)
    r_squared_values.append(r_squared)

# Save the regression coefficients and R-squared value to a new CSV file
regression_summary = pd.DataFrame({
    'Position': positions,
    'Intercept': intercepts,
    'Slope': slopes,
    'R-squared': r_squared_values
})
regression_summary_file_path = os.path.expanduser('~/Desktop/ISP Project/Regression_Summary.csv')
regression_summary.to_csv(regression_summary_file_path, index=False)

print(f"Regression summary file saved to {regression_summary_file_path}")

# Load the updated CSV file 
file_path = os.path.expanduser('~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv')
merged_df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Group by 'Pos' and 'College/Univ', and sum the 'Total Fantasy Points'
grouped = merged_df.groupby(['Pos', 'College/Univ'])['Total Fantasy Points'].sum().compute().reset_index()

# Get the top three schools for each position
top_schools = grouped.groupby('Pos').apply(lambda x: x.nlargest(3, 'Total Fantasy Points')).reset_index(drop=True)

# Calculate the number of players drafted by each school
player_counts = merged_df.groupby(['Pos', 'College/Univ']).size().compute().reset_index(name='Player Count')

# Merge the player counts with the top schools data
top_schools = top_schools.merge(player_counts, on=['Pos', 'College/Univ'])

# Calculate the total number of players drafted for each position
total_player_counts = merged_df.groupby('Pos').size().compute().reset_index(name='Total Player Count')

# Merge the total player counts with the top schools data
top_schools = top_schools.merge(total_player_counts, on='Pos')

# Create the path to save the charts
charts_folder = os.path.expanduser('~/Desktop/ISP Project/Charts')
if not os.path.exists(charts_folder):
    os.makedirs(charts_folder)

# Function to create bar chart for each position
def create_chart(position):
    data = top_schools[top_schools['Pos'] == position].copy()
    
    # Calculate the percentage of players drafted
    data.loc[:, 'Percentage'] = (data['Player Count'] / data['Total Player Count']) * 100
    
    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data['College/Univ'], data['Total Fantasy Points'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(f'Top 3 Schools for {position} by Total Fantasy Points')
    plt.xlabel('School')
    plt.ylabel('Total Fantasy Points')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Annotate the number of players drafted and the percentage
    for bar, player_count, percentage in zip(bars, data['Player Count'], data['Percentage']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{player_count} players ({percentage:.1f}%)', ha='center', va='bottom')
    
    # Save the chart
    chart_path = os.path.join(charts_folder, f'Top_3_Schools_for_{position}.png')
    plt.savefig(chart_path)
    plt.close()
    
    # Print the number of players drafted at each position
    print(f'Number of players drafted for position {position}: {data["Total Player Count"].iloc[0]}')

# Generate charts for each position
positions = top_schools['Pos'].unique()
for position in positions:
    create_chart(position)

# Paths
file_path = os.path.expanduser('~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv')
charts_directory = os.path.expanduser('~/Desktop/ISP Project/Charts')

# Ensure the charts directory exists
os.makedirs(charts_directory, exist_ok=True)

# Load the CSV file from the specified path
merged_df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Ensure the relevant columns are present and correct data types
merged_df['Year'] = pd.to_numeric(merged_df['Year'], errors='coerce')
merged_df['Draft_Rank'] = pd.to_numeric(merged_df['Draft_Rank'], errors='coerce')
merged_df['Fantasy Points Per Game'] = pd.to_numeric(merged_df['Fantasy Points Per Game'], errors='coerce')
merged_df['Pos'] = merged_df['Pos'].astype(str)

# Drop rows with missing values in critical columns
merged_df = merged_df.dropna(subset=['Draft_Rank', 'Fantasy Points Per Game', 'Pos'])

# Calculate correlation between Draft Rank and Fantasy Points Per Game by Position
correlation_df = merged_df.groupby('Pos').apply(lambda x: x[['Draft_Rank', 'Fantasy Points Per Game']].corr().iloc[0, 1], meta=('Correlation', 'f8')).compute().reset_index(name='Correlation')

# Print the correlation results
print("Correlation between Draft Rank and Fantasy Points Per Game by Position:")
print(correlation_df)

# Convert to Pandas DataFrame for plotting
merged_df = merged_df.compute()

# Plot the data with a regression line
plt.figure(figsize=(14, 8))
sns.scatterplot(data=merged_df, x='Draft_Rank', y='Fantasy Points Per Game', hue='Pos', style='Pos', palette='Set1', s=100)
sns.regplot(data=merged_df, x='Draft_Rank', y='Fantasy Points Per Game', scatter=False, color='gray')

# Add text annotations for correlation values
for i in range(len(correlation_df)):
    plt.text(0.95, 0.95 - i*0.05, f"{correlation_df['Pos'].iloc[i]}: r={correlation_df['Correlation'].iloc[i]:.2f}",
             horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

plt.title('Draft Rank vs Fantasy Points Per Game by Position')
plt.xlabel('Draft Rank')
plt.ylabel('Fantasy Points Per Game')
plt.legend(title='Position')
plt.tight_layout()

# Save the plot
plot_path = os.path.join(charts_directory, 'Draft_Rank_vs_Fantasy_Points_Per_Game.png')
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to: {plot_path}")

# Paths
file_path = os.path.expanduser('~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv')
charts_directory = os.path.expanduser('~/Desktop/ISP Project/Charts')

# Ensure the charts directory exists
os.makedirs(charts_directory, exist_ok=True)

# Load the CSV file from the specified path
merged_df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Ensure the relevant columns are present and correct data types
merged_df['Year'] = pd.to_numeric(merged_df['Year'], errors='coerce')
merged_df['Pick'] = pd.to_numeric(merged_df['Pick'], errors='coerce')
merged_df['Fantasy Points Per Game'] = pd.to_numeric(merged_df['Fantasy Points Per Game'], errors='coerce')
merged_df['Pos'] = merged_df['Pos'].astype(str)

# Drop rows with missing values in critical columns
merged_df = merged_df.dropna(subset=['Pick', 'Fantasy Points Per Game', 'Pos'])

# Calculate correlation between Pick and Fantasy Points Per Game by Position
correlation_df = merged_df.groupby('Pos').apply(lambda x: x[['Pick', 'Fantasy Points Per Game']].corr().iloc[0, 1], meta=('Correlation', 'f8')).compute().reset_index(name='Correlation')

# Print the correlation results
print("Correlation between Pick and Fantasy Points Per Game by Position:")
print(correlation_df)

# Convert to Pandas DataFrame for plotting
merged_df = merged_df.compute()

# Plot the data with a regression line
plt.figure(figsize=(14, 8))
sns.scatterplot(data=merged_df, x='Pick', y='Fantasy Points Per Game', hue='Pos', style='Pos', palette='Set1', s=100)
sns.regplot(data=merged_df, x='Pick', y='Fantasy Points Per Game', scatter=False, color='gray')

# Add text annotations for correlation values
for i in range(len(correlation_df)):
    plt.text(0.95, 0.95 - i*0.05, f"{correlation_df['Pos'].iloc[i]}: r={correlation_df['Correlation'].iloc[i]:.2f}",
             horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

plt.title('Pick vs Fantasy Points Per Game by Position')
plt.xlabel('Pick')
plt.ylabel('Fantasy Points Per Game')
plt.legend(title='Position')
plt.tight_layout()

# Save the plot
plot_path = os.path.join(charts_directory, 'Pick_vs_Fantasy_Points_Per_Game.png')
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to: {plot_path}")

# Load the updated CSV file
file_path = os.path.expanduser('~/Desktop/ISP Project/Merged_Rookie_Stats_with_Fantasy_Points.csv')
merged_df = dd.read_csv(file_path, dtype={'Age': 'float64'})

# Ensure the relevant columns are present and correct data types
merged_df['Year'] = pd.to_numeric(merged_df['Year'], errors='coerce')
merged_df['Draft_Rank'] = pd.to_numeric(merged_df['Draft_Rank'], errors='coerce')
merged_df['Pick'] = pd.to_numeric(merged_df['Pick'], errors='coerce')
merged_df['Fantasy Points Per Game'] = pd.to_numeric(merged_df['Fantasy Points Per Game'], errors='coerce')
merged_df['Pos'] = merged_df['Pos'].astype(str)

# Drop rows with missing values in critical columns
merged_df = merged_df.dropna(subset=['Draft_Rank', 'Pick', 'Fantasy Points Per Game', 'Pos'])

# Calculate correlation between Draft Rank and Fantasy Points Per Game by Position
draft_rank_correlation_df = merged_df.groupby('Pos').apply(lambda x: x[['Draft_Rank', 'Fantasy Points Per Game']].corr().iloc[0, 1], meta=('Correlation_Draft_Rank', 'f8')).compute().reset_index(name='Correlation_Draft_Rank')

# Calculate correlation between Pick and Fantasy Points Per Game by Position
pick_correlation_df = merged_df.groupby('Pos').apply(lambda x: x[['Pick', 'Fantasy Points Per Game']].corr().iloc[0, 1], meta=('Correlation_Pick', 'f8')).compute().reset_index(name='Correlation_Pick')

# Print the correlation results
print("Correlation between Draft Rank and Fantasy Points Per Game by Position:")
print(draft_rank_correlation_df)
print("\nCorrelation between Pick and Fantasy Points Per Game by Position:")
print(pick_correlation_df)

# Convert to Pandas DataFrame for plotting
merged_df = merged_df.compute()

# Plot the data with a regression line for Draft Rank
plt.figure(figsize=(14, 8))
sns.scatterplot(data=merged_df, x='Draft_Rank', y='Fantasy Points Per Game', hue='Pos', style='Pos', palette='Set1', s=100)
sns.regplot(data=merged_df, x='Draft_Rank', y='Fantasy Points Per Game', scatter=False, color='gray')

# Add text annotations for correlation values for Draft Rank
for i in range(len(draft_rank_correlation_df)):
    plt.text(0.95, 0.95 - i*0.05, f"{draft_rank_correlation_df['Pos'].iloc[i]}: r={draft_rank_correlation_df['Correlation_Draft_Rank'].iloc[i]:.2f}",
             horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

plt.title('Draft Rank vs Fantasy Points Per Game by Position')
plt.xlabel('Draft Rank')
plt.ylabel('Fantasy Points Per Game')
plt.legend(title='Position')
plt.tight_layout()

# Save the plot for Draft Rank
plot_path_draft_rank = os.path.join(charts_directory, 'Draft_Rank_vs_Fantasy_Points_Per_Game.png')
plt.savefig(plot_path_draft_rank)
plt.close()

# Plot the data with a regression line for Pick
plt.figure(figsize=(14, 8))
sns.scatterplot(data=merged_df, x='Pick', y='Fantasy Points Per Game', hue='Pos', style='Pos', palette='Set1', s=100)
sns.regplot(data=merged_df, x='Pick', y='Fantasy Points Per Game', scatter=False, color='gray')

# Add text annotations for correlation values for Pick
for i in range(len(pick_correlation_df)):
    plt.text(0.95, 0.95 - i*0.05, f"{pick_correlation_df['Pos'].iloc[i]}: r={pick_correlation_df['Correlation_Pick'].iloc[i]:.2f}",
             horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

plt.title('Pick vs Fantasy Points Per Game by Position')
plt.xlabel('Pick')
plt.ylabel('Fantasy Points Per Game')
plt.legend(title='Position')
plt.tight_layout()

# Save the plot for Pick
plot_path_pick = os.path.join(charts_directory, 'Pick_vs_Fantasy_Points_Per_Game.png')
plt.savefig(plot_path_pick)
plt.close()

print(f"Plots saved to: {plot_path_draft_rank} and {plot_path_pick}")
