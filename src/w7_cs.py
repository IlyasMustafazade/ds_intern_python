import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster as skl_cluster
from sklearn import decomposition

import cluster_reporter
import dendrogram
import kmeans
import point_array_extractor
import silhouette
from funcs import *


def main():
    file_name = "Country-data"
    file_ext = ".csv"
    full_name = file_name + file_ext
    df = pd.read_csv(full_name)
    country_arr = np.array(df["country"])
    df = df.drop(labels="country", axis=1)
    df_columns = df.columns
    to_normalize = df_columns[1:]
    df = normalize(
        f_space=df, col_arr=to_normalize)
    observation_set = df.values
    point_obj_arr = point_array_extractor.PointArrayExtractor.extract_point_arr(
        observation_set=observation_set)
    # KMEANS CLUSTERING WITH PCA
    pca = decomposition.PCA(n_components=2)
    pca.fit(observation_set)
    observation_set = pca.transform(
        observation_set)
    distance_metric = "euclidean"
    n_clusters = 3
    my_kmeans_clusterer = kmeans.KMeans(
        observation_set=observation_set, max_iter=300, distance_metric=distance_metric,
        n_clusters=n_clusters)
    kmeans_clusterer = skl_cluster.KMeans(
        n_clusters=n_clusters)
    my_kmeans_clusterer.fit()
    kmeans_clusterer.fit(observation_set)
    cluster_reporter.ClusterReporter.report(n_clusters=n_clusters, clusterer=my_kmeans_clusterer,
                                            clusterer_name="Sklearn KMeans",
                                            observation_set=observation_set,
                                            column=country_arr, distance_metric=distance_metric,
                                            ground_truth_exists=False)
    cluster_reporter.ClusterReporter.report_skl(n_clusters=n_clusters, clusterer=kmeans_clusterer,
                                                clusterer_name="Sklearn KMeans",
                                                observation_set=observation_set,
                                                column=country_arr,
                                                ground_truth_exists=False)
    # HIERARCHICAL CLUSTERING WITH PCAsss
    hierarchical_clusterer = skl_cluster.AgglomerativeClustering(
        n_clusters=n_clusters, distance_threshold=None, compute_distances=True)
    hierarchical_clusterer.fit(observation_set)
    cluster_reporter.ClusterReporter.report_skl(n_clusters=n_clusters, clusterer=hierarchical_clusterer,
                                                clusterer_name="Hierarchical", observation_set=observation_set, column=country_arr)
    dendrogram.Dendrogram.plot_dendrogram(
        model=hierarchical_clusterer, truncate_mode="level", p=3)
    plt.show()


if __name__ == "__main__":
    main()
