import time
import argparse
import sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import metrics


def show_info(text):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("[%s] %s." % (now, text))


def get_data(data_name):
    if data_name == "iris":
        dataset = load_iris()
        show_info("dataset: iris | shape: " + str(dataset.data.shape))
    elif data_name == "digits":
        dataset = load_digits()
        show_info("dataset: digits | shape: " + str(dataset.data.shape))
    else:
        show_info("no such dataset [%s]" % data_name)
        return None, None
    return dataset.data, dataset.target


def preprocess_data(x, std=None):
    if std == "zscore":
        x_s = preprocessing.scale(x)
        show_info("data standardization by z-score")
    elif std == "minmax":
        x_s = preprocessing.minmax_scale(x)
        show_info("data standardization by min-max")
    else:
        show_info("no standardization")
        return x
    return x_s


def reduce_dim(x, dim, method="pca"):
    if dim <= 0 or dim >= x.shape[1]:
        return  x
    if method == "pca":
        reduce = PCA(n_components=dim)
        show_info("reduce dimension by PCA")
        x_r = reduce.fit(x).transform(x)
    else:
        return x
    return x_r


def main(args):
    x, y = get_data(args.data)
    m = len(np.unique(y))
    assert x is not None and y is not None

    x_s = preprocess_data(x, std=args.standardization)
    x_r = reduce_dim(x_s, dim=args.decomposition)

    if args.algorithm == "svm":
        show_info("classification by svm")
        x_train, x_test, y_train, y_test = train_test_split(x_r, y, random_state=1, train_size=0.7)
        classifier = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
        classifier.fit(x_train, y_train.ravel())
        # y_pred_train = classifier.predict(x_train)
        y_pred_test = classifier.predict(x_test)
        show_info('svm classification result | accuracy: %.2f %%' % (np.mean(y_pred_test == y_test) * 100))
    elif args.algorithm == "knn":
        show_info("classification by knn")
        x_train, x_test, y_train, y_test = train_test_split(x_r, y, random_state=1, train_size=0.7)
        classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
        classifier.fit(x_train, y_train.ravel())
        # y_pred_train = classifier.predict(x_train)
        y_pred = classifier.predict(x_test)
        show_info('knn classification result | accuracy: %.2f %%' % (np.mean(y_pred == y_test) * 100))
    elif args.algorithm == "kmeans":
        show_info("clustering by k-means")
        estimator = KMeans(init='k-means++', n_clusters=m, n_init=m)
        estimator.fit(x_r)
        y_pred = estimator.labels_
        show_info("k-means clustering result | ARI: %.3f | AMI: %.3f" % (metrics.adjusted_rand_score(y, y_pred),
                                                                 metrics.adjusted_mutual_info_score(y, y_pred)))
    else:
        show_info("no %s algorithm now" % args.algorithm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='traditional machine learning algorithm')
    parser.add_argument('--data', type=str, default="iris")
    parser.add_argument('--standardization', type=str, default=None)
    # parser.add_argument('--normalization', type=str, default=None)
    parser.add_argument('--decomposition', type=int, default=2)
    parser.add_argument('--algorithm', type=str, default=None)
    # parser.add_argument('--output', type=str, default="./")
    args = parser.parse_args()

    main(args)
