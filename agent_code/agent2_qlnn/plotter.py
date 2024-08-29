import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_scores(csv_file, miny, maxy, auto_update=False, update_interval=5000, smoothing_window=10):
    scores = pd.read_csv(csv_file)
    scores_smooth = scores.rolling(window=smoothing_window).mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up the initial plot
    ax.set_title('Scores Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Score')
    ax.grid(True)

    line, = ax.plot(range(1, len(scores_smooth) + 1), scores_smooth, linestyle='-', color='b')
    ax.set_xlim(smoothing_window, len(scores_smooth) + 1)
    ax.set_ylim(miny, maxy)

    def update(frame):
        scores = pd.read_csv(csv_file)
        scores_smooth = scores.rolling(window=smoothing_window).mean()

        # Update the data for the line plot
        line.set_data(range(1, len(scores_smooth) + 1), scores_smooth)
        ax.set_xlim(smoothing_window, len(scores_smooth) + 1)

        return line,

    if auto_update:
        # Create the animation
        ani = animation.FuncAnimation(fig, update, interval=update_interval)

    plt.show()

plot_scores('model/scores.csv', miny=0, maxy=30, auto_update=True)
