import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('model/training_metrics.csv')

print("Columns in CSV:", data.columns)

epochs = data['Round']
scores = data['Score']
survivals = data['Survived']
cumulative_rewards = data['CumulativeReward']
kills = data['Kills']
suicides = data['Suicides']
opponents_eliminated = data['OpponentsEliminated']


def smooth_data(series, ws):
    return series.rolling(window=ws, min_periods=1).mean()


window_sizes = [5, 15, 30]
fig, ax = plt.subplots(6, 1, figsize=(10, 15), sharex=True)

# Plot Score
for window_size in window_sizes:
    ax[0].plot(epochs, smooth_data(scores, window_size), label=f'Smoothed (window={window_size})')
ax[0].set_ylabel('Score')
ax[0].set_title('Training Metrics over Epochs')
ax[0].legend()

# Plot Survival
for window_size in window_sizes:
    ax[1].plot(epochs, smooth_data(survivals, window_size), label=f'Smoothed (window={window_size})')
ax[1].set_ylabel('Survived')
ax[1].legend()

# Plot Cumulative Reward
for window_size in window_sizes:
    ax[2].plot(epochs, smooth_data(cumulative_rewards, window_size), label=f'Smoothed (window={window_size})')
ax[2].set_ylabel('Cumulative Reward')
ax[2].legend()

# Plot Kills
for window_size in window_sizes:
    ax[3].plot(epochs, smooth_data(kills, window_size), label=f'Smoothed (window={window_size})')
ax[3].set_ylabel('Kills')
ax[3].legend()

# Plot Opponents Eliminated
for window_size in window_sizes:
    ax[4].plot(epochs, smooth_data(opponents_eliminated, window_size), label=f'Smoothed (window={window_size})')
ax[4].set_ylabel('Opponents Eliminated')
ax[4].set_xlabel('Epoch')
ax[4].legend()

# Plot Suicides
for window_size in window_sizes:
    ax[5].plot(epochs, smooth_data(suicides, window_size), label=f'Smoothed (window={window_size})')
ax[5].set_ylabel('Suicides')
ax[5].set_xlabel('Epoch')
ax[5].legend()

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()
