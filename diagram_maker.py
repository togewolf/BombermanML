import json
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the JSON data from a file
# Replace 'data.json' with the path to your JSON file
with open('results/thomas_vs_crow.json', 'r') as file:
    data = json.load(file)

# Step 2: Extract the metrics and convert them to a DataFrame
# 'by_agent' contains all the agents and their respective metrics
df = pd.DataFrame(data['by_agent']).T

# List of metrics to plot
metrics = df.columns

# Step 3: Create bar charts for each metric comparing the agents
for metric in metrics:
    plt.figure(figsize=(10, 6))
    df[metric].plot(kind='bar')
    plt.title(f'Comparison of {metric.capitalize()} by Agent')
    plt.xlabel('Agent')
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()