from ast import parse
from pickle import load
from scipy.cluster.vq import kmeans, vq, whiten, kmeans2
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import argparse


def KMEANS(args):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    
    # whiten datas to reduce the relation among features and to set the cov to be 1
    X_train = whiten(X_train)
    X_test = whiten(X_test)

    # kmeans
    # codebook: A k by N array of k centroids(中心点)
    # distortion: The mean Euclidean distance between observations and centroids
    codebook, distortion = kmeans(X_train, k_or_guess=args.k, iter=args.max_iter)

    # vq
    # code: A length M array holding the code book index for each observation(观测类)
    # dist: The distortion (distance) between the observation and its nearest code
    code, dist = vq(X_train, codebook)

    print("Centroids:")
    print(codebook)
    print("Labels:")
    print(code)

    # slice = [[0,1,3,6,7,...], [2,4,8,9,...], [5,10,12,15,...]]
    slice = get_slice(code, args.k)
    for i in range(args.k):
        print("Cluster {}: <{} points>".format(i, len(slice[i])))
        print(slice[i])
        plt.scatter(X_train[slice[i], 0], X_train[slice[i], 1])
        plt.scatter(codebook[i, 0], codebook[i, 1], c="red", marker="*", linewidths=2)
    plt.show()


def KMEANS2(args):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    X_train = whiten(X_train)
    X_test = whiten(X_test)

    # kmeans2将工作集中在一个方法中：返回中心点和观测标签
    centroid, label = kmeans2(X_train, args.k, iter=args.max_iter)
    print("Centroids:")
    print(centroid)
    print("Labels:")
    print(label)

    slice = get_slice(label, args.k)
    for i in range(args.k):
        print("Cluster {}: <{} points>".format(i, len(slice[i])))
        print(slice[i])
        plt.scatter(X_train[slice[i], 0], X_train[slice[i], 1])
        plt.scatter(centroid[i, 0], centroid[i, 1], c="red", marker="*", linewidths=2)
    plt.show()


def HIERARCHICAL(args):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.8)
    X_train = whiten(X_train)
    X_test = whiten(X_test)

    # Z: The hierarchical clustering encoded as a linkage matrix
    Z = linkage(X_train, method="ward")
    # R: A dictionary of data structures computed to render the dendrogram
    R = dendrogram(Z)
    # print(R)
    plt.show()


def get_slice(labels, k):
    slice = [[] for _ in range(k)]
    for index, label in enumerate(labels):
        slice[label].append(index)
    return slice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--method", type=str, default="hierarchical", help="hierachical, kmeans, or kmeans2")
    args = parser.parse_args()

    if args.method == "hierarchical":
        HIERARCHICAL(args)
    elif args.method == "kmeans":
        KMEANS(args)
    elif args.method == "kmeans2":
        KMEANS2(args)
    else:
        raise Exception("Method must be chosen from hierachical, kmeans, or kmeans2!")
