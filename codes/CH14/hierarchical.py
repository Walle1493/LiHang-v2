from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import argparse
from clustering import ClusterAgglomerative


def main(args):
    iris = load_iris()
    data = iris.data
    clustering = ClusterAgglomerative(k=args.k, maxiter=args.maxiter)
    clustering.fit(data)

    gs = clustering.gs
    for idx, gs_ in enumerate(gs):
        print("<Clustering {}>".format(idx + 1))
        # print(">Name:", gs_.name)
        print("--Amount:", len(gs_.data))
        plt.scatter(gs_.data[:, 0], gs_.data[:, 1])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=1000)
    args = parser.parse_args()
    main(args)

# <Clustering 1>
# --Amount: 50
# <Clustering 2>
# --Amount: 100
