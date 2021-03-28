import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import plotly.graph_objects as go


def plot_losses(train_losses, test_losses, k, title):
    plt.plot(np.arange(0, len(train_losses)), train_losses, '-b', label='Train Loss')
    plt.plot(np.arange(0, len(test_losses)), test_losses, '-r', label='Test Loss')

    plt.xlabel('Nr. iterations')
    plt.legend(loc='upper right')
    plt.title(title)

    # save image
    plt.savefig(title + ".png")  # should before show method

    # plt.show()
    plt.close()


def plot_confusion_matrix(confusion_matrix, title):
    plt.figure(figsize = (10,7))
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
    sn.heatmap(df_cm, annot=True)

    plt.savefig(f"{title}.png")
    plt.close()


def plot_metrics_table(precision, recall, f_score, title):
    header_data = ["Classes", "Precision", "Recall", "F-score"]

    data = [np.arange(precision.shape[0]).T, precision.T, recall.T, f_score.T]

    fig = go.Figure(data=[go.Table(header=dict(values=header_data),
                    cells=dict(values=data))
                        ])
    
    fig.write_image(f"{title}.png")
    
    # fig.show()