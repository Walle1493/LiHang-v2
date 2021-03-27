from ast import parse
from pydoc import cli
import re
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering, KMeans
import argparse

from sklearn.utils.sparsefuncs import inplace_csr_column_scale


def hierarchical(args):
    iris = load_iris()
    data = iris.data
    clustering = AgglomerativeClustering(n_clusters=args.n_clusters, linkage=args.linkage)
    clustering.fit(data)

    n_clusters = clustering.n_clusters_
    labels = clustering.labels_
    print("Clusters:", n_clusters)
    print("Labels:", labels)
    print()

    # get cluster label for each data
    slice = get_slice(labels, n_clusters)
    # print(slice)

    for i in range(n_clusters):
        print("Cluster {}: <{} points>".format(i, len(slice[i])))
        print(slice[i])
        plt.scatter(data[slice[i], 0], data[slice[i], 1])
    plt.show()


def kmeans(args):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    clustering = KMeans(n_clusters=args.n_clusters, n_init=args.n_init, max_iter=args.max_iter)
    clustering.fit(X_train, y_train)
    # or directly use
    # clustering.fit(X_train)
    # clustering.fit_transform(X_train, y_train)

    labels = clustering.labels_
    n_clusters = len(set(labels))
    print("Clusters:", n_clusters)
    print("Labels:", labels)
    print()

    slice = get_slice(labels, n_clusters)

    for i in range(n_clusters):
        print("Cluster {}: <{} points>".format(i, len(slice[i])))
        print(slice[i])
        plt.scatter(X_train[slice[i], 0], X_train[slice[i], 1])
    plt.show()

    # TODO: predict the X_test datasets and calculate the accuracy


def get_slice(labels, clusters):
    slice = [[] for _ in range(clusters)]
    for index, label in enumerate(labels):
        slice[label].append(index)
    return slice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--linkage", type=str, default="ward", help="['ward', 'complete', 'average', 'simgle']")
    parser.add_argument("--n_init", type=int, default=10, help="Number of time the k-means algorithm will be run with different centroid seeds.")
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--method", type=str, default="kmeans", help="kmeans or hierarchical")
    args = parser.parse_args()

    if args.method == "kmeans":
        print("Use K-Means...")
        kmeans(args)
    else:
        print("Use Hierarchical Clustering...")
        hierarchical(args)
