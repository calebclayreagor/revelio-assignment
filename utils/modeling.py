import numpy as np
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN

def scale_local_density(X: np.array,
                        w: np.array,
                        gamma: float,
                        n_nbrs: int = 10,
                        eps = 1e-6
                        ) -> np.array:
    nbrs = NearestNeighbors(n_neighbors = n_nbrs).fit(X)
    d, ix = nbrs.kneighbors(X)
    d_nbr = d[:, 1:]
    w_nbr = w[ix[:, 1:]]
    dens = (w_nbr / (d_nbr + eps)).mean(1)
    return X * dens.reshape(-1, 1) ** gamma


def fit_model(gdf: gpd.GeoDataFrame,
              clf_name: str,
              feature_keys: list[str] = ['x', 'y'],
              weight_key: str = 'w',
              scale_features: bool = True,
              gamma_density: float | None = None,
              eps_DBSCAN: float | None = None,
              min_samples_DBSCAN: int | None = None,
              min_cluster_size_HDBSCAN: int | None = None,
              min_samples_HDBSCAN: int | None = None,
              cluster_selection_epsilon_HDBSCAN: float | None = None,
              ) -> gpd.GeoDataFrame:

    gdf = gdf.copy()
    X = gdf[feature_keys].values
    w = gdf[weight_key].values

    if scale_features:
        X = scale_local_density(X, w, gamma_density)

    if clf_name == 'DBSCAN':
        clf = DBSCAN(eps = eps_DBSCAN, min_samples = min_samples_DBSCAN)
        gdf['cluster'] = clf.fit_predict(X, sample_weight = w)
   
    elif clf_name == 'HDBSCAN':
        clf = HDBSCAN(min_cluster_size = min_cluster_size_HDBSCAN,
                      min_samples = min_samples_HDBSCAN,
                      cluster_selection_epsilon = cluster_selection_epsilon_HDBSCAN)
        gdf['cluster'] = clf.fit_predict(X)

    # unassign singleton clusters
    counts = gdf.cluster.value_counts()
    gdf['cluster'] = gdf.cluster.where(gdf.cluster.map(counts) > 1, -1)
    return gdf


