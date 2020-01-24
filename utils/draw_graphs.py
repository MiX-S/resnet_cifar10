import matplotlib.pyplot as plt


def acc_loss_graph(accuracies, losses):
    """
    Draw accuracies and losses for every net on 1 figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for experiment_id in accuracies.keys():
        ax1.plot(accuracies[experiment_id], label=experiment_id)
    ax1.legend()
    ax1.set_title('Validation Accuracy')
    fig.tight_layout()

    for experiment_id in accuracies.keys():
        ax2.plot(losses[experiment_id], label=experiment_id)
    ax2.legend()
    ax2.set_title('Validation Loss');

    fig.tight_layout()