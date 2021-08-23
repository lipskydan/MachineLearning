import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron
from plottingDecisionRegions import plot_decision_regions


def main():
    """Read dataset of iris from iris.data"""
    df = pd.read_csv('../iris.data', header=None)
    df.tail()

    """Select first 100 elements from dataset
    1-50 are setosa
    51-100 are versicolor
    Marked these types as -1 and 1   
    """
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    """Extract to columns - sepal length and petal length"""
    X = df.iloc[0:100, [0, 2]].values

    """Illustrate dataset"""
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='x', label='versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    # plt.savefig('images/extract_dataset.png', dpi=300)
    # plt.clf()
    plt.show()

    """Training the perceptron model"""
    ppn = Perceptron(eta=0.1, n_iter=10)

    ppn.fit(X, y)

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')

    # plt.savefig('images/convergence_graph.png', dpi=300)
    # plt.clf()
    plt.show()

    """Illustrate a decision regions"""
    plot_decision_regions(X, y, classifier=ppn, label_f='setosa', label_s='versicolor')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    # plt.savefig('images/perceptron_result.png', dpi=300)
    # plt.clf()
    plt.show()


if __name__ == "__main__":
    main()
