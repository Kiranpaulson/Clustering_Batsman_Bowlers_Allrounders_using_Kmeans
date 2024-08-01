import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
def cleancsv():
       

    # Read the CSV file into a pandas DataFrame
    input_file = 'players.csv'
    df = pd.read_csv(input_file)

    # Drop rows with at least one null value
    df_cleaned = df.dropna()

    # Define a function to remove outliers based on IQR
    def remove_outliers_iqr(data_frame, column):
        Q1 = data_frame[column].quantile(0.25)
        Q3 = data_frame[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data_frame[(data_frame[column] >= lower_bound) & (data_frame[column] <= upper_bound)]

    # Specify columns for which you want to remove outliers using IQR
    columns_to_remove_outliers = ['BattingAVG', 'EconomyRate','BattingS/R', 'BowlingAVG']

    # Apply the remove_outliers_iqr function to specified columns
    for column in columns_to_remove_outliers:
        df_cleaned = remove_outliers_iqr(df_cleaned, column)

    # Save the cleaned DataFrame to a new CSV file after removing outliers
    output_file = 'cleaned_output_file_with_outliers_removed.csv'
    df_cleaned.to_csv(output_file, index=False)

    print(f"Rows with null values and outliers removed. Cleaned data saved to {output_file}.")
 

def read_csv_file(file_path):
    player_names = []

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)

        # Assuming the first row contains headers (Player, BattingAverage, BowlingEconomy, StrikeRate, Wickets)
        headers = next(csv_reader)

        # Assuming the player names are in the first column
        for row in csv_reader:
            player_names.append(row[0])

    return player_names


def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def k_means_clustering(data, k, max_iterations=100):
    # Use the first k elements as the initial centroids
    centroids = [[30.65, 12.00, 137.97, 29.66], [3.95, 8.32, 51.36, 30.84], [10.97, 8.22, 100.73, 29.94]]

#completely changed trail and errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    print(centroids)
    for _ in range(max_iterations):
        # Assign each data point to the closest centroid
        labels = [min(range(k), key=lambda i: euclidean_distance(point, centroids[i])) for point in data]

        # Update centroids based on the mean of the assigned data points
        new_centroids = [
            [
                sum(data[j][dim] for j in range(len(data)) if labels[j] == i) / labels.count(i)
                for dim in range(len(data[0]))
            ]
            for i in range(k)
        ]

        print(f"\nIteration{_ + 1} clusters:")
        for i in range(k):
            cluster_indices = [j for j in range(len(data)) if labels[j] == i]
            cluster_name = "batsman" if i == 0 else ("bowlers" if i == 1 else "allrounders")
            players_in_cluster = [(data[j], player_names[j]) for j in cluster_indices]
            print(f"{cluster_name}: {players_in_cluster}")

        # Check for convergence
        if new_centroids == centroids:
            break

        centroids = new_centroids

    return labels, centroids

def print_cluster_table(cluster_name, player_names):
    print(f"\n{cluster_name}\n------------")
    for player in player_names:
        print(player)


cleancsv()
# Specify the path to your CSV file
csv_file_path = 'cleaned_output_file_with_outliers_removed.csv'#only the names of the file has been chagen here for the data from KLEIGG
#now i have changed it to both claened output after removal of null and outlier
# Initialize an empty list to store the data
data = []

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'cleaned_output_file_with_outliers_removed.csv'
player_names = read_csv_file(file_path)

# Open the CSV file
with open(csv_file_path, 'r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)

    # Skip the header row
    header = next(csv_reader)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Convert values to float, excluding the first column
        float_row = [float(value) for value in row[1:]]
        data.append(float_row)

# Get user input for the number of clusters (k)
k = int(input("Enter the number of clusters (k): "))

# Perform k-means clustering
labels, centroids = k_means_clustering(data, k)


print("\nFinal centroids:")
print(centroids)

# Display results
print("\nFinal clusters:")

for i in range(k):
    cluster_indices = [j for j in range(len(data)) if labels[j] == i]
    cluster_name = "batsman" if i == 0 else ("bowlers" if i == 1 else "allrounders")
    players_in_cluster = [(player_names[j]) for j in cluster_indices]
    print(f"{cluster_name}: {players_in_cluster}")
#edited on 30-11-2023
for i in range(k):
    cluster_indices = [j for j in range(len(data)) if labels[j] == i]
    cluster_name = "BATSMEN" if i == 0 else ("BOWLERS" if i == 1 else "ALL-ROUNDERS")
    players_in_cluster = [player_names[j] for j in cluster_indices]

    # Print the cluster table
    print_cluster_table(cluster_name, players_in_cluster)



    
# After performing k-means clustering
# ...

# Create a dictionary to store players and their corresponding cluster labels
player_clusters = {player_names[i]: labels[i] for i in range(len(player_names))}

# ... (previous code remains unchanged)

# Create dictionaries to store performance metrics for each cluster
batsman_performance = {}
bowler_performance = {}
allrounder_performance = {}

# Iterate over players and their clusters
for player, cluster_label in player_clusters.items():
    player_data = data[player_names.index(player)]
    
    # Extract relevant performance metrics
    batting_avg = player_data[0]  # Batting Average
    economy_rate = player_data[1]  # Economy Rate
    strike_rate = player_data[2]  # Strike Rate
    bowling_avg = player_data[3]  # Bowling Average
    

    # Update the performance dictionary based on the cluster label
    if cluster_label == 0:  # Batsman
        performance_score = 0.6 * batting_avg + 0.4 * strike_rate
        batsman_performance[player] = performance_score

    elif cluster_label == 1:  # Bowler
        performance_score = -0.6 * bowling_avg - 0.4 * economy_rate  # Negative for sorting in ascending order
        bowler_performance[player] = performance_score

    else:  # All-rounder
        performance_score = 0.5 * (0.6 * batting_avg + 0.4 * strike_rate) - 0.5 * (0.6 * bowling_avg + 0.4 * economy_rate)
        allrounder_performance[player] = performance_score

# Sort players based on performance in each category
top_batsmen = sorted(batsman_performance, key=batsman_performance.get, reverse=True)[:5]
top_bowlers = sorted(bowler_performance, key=bowler_performance.get)[:5]
top_allrounders = sorted(allrounder_performance, key=allrounder_performance.get, reverse=True)[:5]

# Display the top players
print("\nTop 5 Batsmen:")
for i, player in enumerate(top_batsmen, 1):
    print(f"{i}. {player} - Performance Score: {batsman_performance[player]}")

print("\nTop 5 Bowlers:")
for i, player in enumerate(top_bowlers, 1):
    print(f"{i}. {player} - Performance Score: {bowler_performance[player]}")

print("\nTop 5 All-rounders:")
for i, player in enumerate(top_allrounders, 1):
    print(f"{i}. {player} - Performance Score: {allrounder_performance[player]}")




# Create a dictionary to store players and their corresponding cluster labels
player_clusters = {player_names[i]: labels[i] for i in range(len(player_names))}



# Create scatter plots for each cluster
for i in range(k):
    cluster_indices = [j for j in range(len(data)) if labels[j] == i]
    cluster_name = "batsman" if i == 0 else ("bowlers" if i == 1 else "allrounders")

    # Extract x and y coordinates for the scatter plot
    x_values = [data[j][0] for j in cluster_indices]  # Assuming Batting Average for x-axis
    y_values = [data[j][2] for j in cluster_indices]  # Assuming Strike Rate for y-axis

    # Plot the scatter plot
    plt.scatter(x_values, y_values, label=f"{cluster_name} Cluster")

# Plot the centroids
centroids_x = [centroid[0] for centroid in centroids]
centroids_y = [centroid[2] for centroid in centroids]
plt.scatter(centroids_x, centroids_y, marker='X', s=200, color='red', label='Centroids')

# Set plot labels and legend
plt.xlabel('Bowling average')
plt.ylabel('Strike Rate')
plt.title('K-Means Clustering of Cricket Players')
plt.legend()
plt.grid(True)


plt.show()
