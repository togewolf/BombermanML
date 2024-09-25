import json
import pandas as pd
import matplotlib.pyplot as plt

with open('results/thomas_vs_crow.json', 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data['by_agent']).T

metrics = df.columns

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