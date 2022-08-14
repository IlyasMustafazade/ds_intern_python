import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster as skl_cluster
from sklearn import metrics

import cluster_reporter
import dendrogram
import kmeans
import point_array_extractor
from funcs import *


def main():
    file_name = "Iris"
    file_ext = ".csv"
    full_name = file_name + file_ext
    df = pd.read_csv(full_name)
    df = df.drop(labels=["Id"], axis=1)
    species_arr = df["Species"]
    species_arr = np.array(species_arr)
    df = df.drop(labels=["Species"], axis=1)
    observation_set = df.values
    point_arr = point_array_extractor.PointArrayExtractor.extract_point_arr(
        observation_set=observation_set)
    # KMEANS CLUSTERING
    distance_metric = "euclidean"
    n_clusters = 3
    kmeans_clusterer = kmeans.KMeans(max_iter=300,
                                     distance_metric=distance_metric, n_clusters=n_clusters,
                                     observation_set=observation_set)
    kmeans_clusterer.fit()
    cluster_reporter.ClusterReporter.report(n_clusters=n_clusters, clusterer=kmeans_clusterer,
                                            clusterer_name="KMeans",
                                            observation_set=observation_set,
                                            column=species_arr, distance_metric=distance_metric,
                                            ground_truth_exists=True)
    skl_kmeans_clusterer = skl_cluster.KMeans(
        n_clusters=n_clusters)
    skl_kmeans_clusterer.fit(observation_set)
    cluster_reporter.ClusterReporter.report_skl(n_clusters=n_clusters, clusterer=skl_kmeans_clusterer,
                                                clusterer_name="Sklearn KMeans",
                                                observation_set=observation_set,
                                                column=species_arr, ground_truth_exists=True)
    # HIERARCHICAL CLUSTERING
    hierarchical_clusterer = skl_cluster.AgglomerativeClustering(
        n_clusters=n_clusters, distance_threshold=None, compute_distances=True)
    hierarchical_clusterer.fit(observation_set)
    cluster_reporter.ClusterReporter.report_skl(n_clusters=n_clusters, clusterer=hierarchical_clusterer,
                                                clusterer_name="Hierarchical", observation_set=observation_set,
                                                column=species_arr, ground_truth_exists=True)
    dendrogram.Dendrogram.plot_dendrogram(
        model=hierarchical_clusterer, truncate_mode="level", p=3)
    plt.show()


if __name__ == "__main__":
    main()
