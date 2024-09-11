import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_scores_and_survival(score_file, survival_file, miny_score, maxy_score, miny_survival, maxy_survival,
                             auto_update=False, update_interval=5000, smoothing_window=10, view_range=10000):
    # Load data
    scores = pd.read_csv(score_file)
    survival = pd.read_csv(survival_file)

    # Smooth data
    scores_smooth = scores.rolling(window=smoothing_window).mean()
    survival_smooth = survival.rolling(window=smoothing_window).mean()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Set up the first y-axis for scores
    ax1.set_title('Scores and Survival Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Score', color='b')
    ax1.grid(True)

    # Plot scores
    line1, = ax1.plot(range(1, len(scores_smooth) + 1), scores_smooth, linestyle='-', color='b', label="Scores")
    ax1.set_xlim(max(smoothing_window, len(scores_smooth) - view_range), len(scores_smooth) + 1)
    ax1.set_ylim(miny_score, maxy_score)

    # Create a second y-axis for survival on the same plot
    ax2 = ax1.twinx()
    ax2.set_ylabel('Survival', color='g')

    # Plot survival
    line2, = ax2.plot(range(1, len(survival_smooth) + 1), survival_smooth, linestyle='-', color='g', label="Survival")
    ax2.set_ylim(miny_survival, maxy_survival)

    # Update function for animation
    def update(frame):
        # Reload and smooth data for scores and survival
        scores = pd.read_csv(score_file)
        survival = pd.read_csv(survival_file)
        scores_smooth = scores.rolling(window=smoothing_window).mean()
        survival_smooth = survival.rolling(window=smoothing_window).mean()

        # Update the data for the line plots
        line1.set_data(range(1, len(scores_smooth) + 1), scores_smooth)
        line2.set_data(range(1, len(survival_smooth) + 1), survival_smooth)

        # Update the x-axis limits
        ax1.set_xlim(max(smoothing_window, len(scores_smooth) - view_range), len(scores_smooth) + 1)

        return line1, line2

    if auto_update:
        # Create the animation
        ani = animation.FuncAnimation(fig, update, interval=update_interval)

    plt.show()


# Call the function
plot_scores_and_survival('model/scores.csv', 'model/survivals.csv',
                         miny_score=0, maxy_score=50,
                         miny_survival=0, maxy_survival=1,
                         auto_update=True, update_interval=5000, smoothing_window=10)
