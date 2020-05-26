from pandas import DataFrame
from matplotlib import pyplot as plt
import numpy as np

def plot_dataset(X, Y, m = None, c = 0):
    """
    Plot 2 dimensional generated data with decision boundary for testing SVM

    :param numpy.array X : Input data
    :param numpy.array Y : Labels
    :param numpy.array m : weights learned by SVM
    :param float c : bias learned by SVM
    """         
    df = DataFrame(dict(x=X[0,:], y=X[1,:], label=Y[0, :]))
    colors = {0:'red', 1:'blue'}
    fig, ax = plt.subplots()    
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    if m is not None:
        m = m.flatten()
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())        
        y_vals = -(c + m[0] * x_vals)/m[1]
        plt.plot(x_vals, y_vals, '--')
    plt.show()