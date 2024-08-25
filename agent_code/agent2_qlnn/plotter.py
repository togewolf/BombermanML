import pandas as pd
import matplotlib.pyplot as plt


def plot_scores(scores):
    scores_smooth = scores.rolling(window=100).mean()

    time = range(1, len(scores_smooth) + 1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, scores_smooth, linestyle='-', color='b')
    plt.title('Scores Over Time')
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()


scores = pd.read_csv('scores.csv')
plot_scores(scores)
