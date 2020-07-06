from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from pandas import DataFrame
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from sklearn.decomposition import PCA


def classify_by_clusters(training_data: DataFrame, testing_data: DataFrame, n_clusters: int, cluster_label):
    """
        - create clustering models and train in separate process that then
        - classifies data according to model and returns the classified data somehow
        - TODO group by era and learn to predict which era, cluster eras etc.
        - TODO Group by era and learn to predict which era

    Args:
        training_data (DataFrame): The training data, including only features and target. (only use features though...)
        testing_data (DataFrame): The testing data, including only features and target. (only use features though...)
        n_clusters (int): The number of clusters to train into.
        cluster_label: the name of the column to add cluster labels to.

    Returns:
        bool: The return value. True for success, False otherwise.

    """
    print("Training model")

    show_kmeans(training_data, testing_data, n_clusters, cluster_label)


def show_kmeans(training_data: DataFrame, testing_data: DataFrame, n_clusters: int, cluster_label):
    training_data["erano"] = training_data.era.str.slice(3).astype(int)
    eras = training_data.erano
    usecols = [x for x in training_data.keys() if x.startswith(('feature'))]

    by_era = training_data.groupby(eras).apply(lambda x: averages(x, usecols))

    k_means = KMeans(n_clusters=n_clusters).fit(by_era[usecols].values)
    pickle.dump(k_means, open(str(n_clusters) + "_kmeans.p", "wb"))

    training_data[cluster_label], testing_data[cluster_label] = k_means.predict(
        training_data[usecols]), k_means.predict(testing_data[usecols])


def averages(era_values: DataFrame, usecols):
    averages = era_values[usecols].mean(axis=0, skipna=True)
    return averages


def create_PCA(by_era, usecols, n):
    pca = PCA(n_components=n)
    fit = pca.fit(by_era[usecols])
    print("Explained Variance: %s" % fit.explained_variance_ratio_)
    X_pca = pca.transform(by_era[usecols])
    return X_pca


def create_linkage_tree(X_pca, n_eras):
    linked = linkage(X_pca, method='ward')  # other methods are centroid and single

    label_list = [i for i in range(n_eras)]

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               labels=label_list,
               distance_sort='descending',
               show_leaf_counts=True)
    plt.show()


# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse
