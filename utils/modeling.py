import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm.notebook import tqdm
from sklearn.cluster import DBSCAN
from utils.evaluate import evaluate_density

def fit_iterative_density_model(gdf: gpd.GeoDataFrame,
                                eps_dbscan: float,
                                min_samples_dbscan: int,
                                min_samples_decay: float = .98,
                                n_iter: int = 200,
                                return_metrics: bool = False,
                                return_gdf: bool = False,
                                cluster_key: str = 'cluster',
                                population_key: str = 'population',
                                name_keys: str | list[str] = ['city', 'state', 'country'],
                                metric_key: str = 'log10_dpc',
                                crs_proj: str = 'EPSG:4087',
    ) -> None | np.ndarray | gpd.GeoDataFrame | tuple[np.ndarray, gpd.GeoDataFrame]:
    """Iteratively applies DBSCAN to unclustered cities"""

    crs = gdf.crs
    gdf = gdf.copy()
    if return_metrics:
        ix_clus, ix_other = 0, 1
        metrics = np.full((n_iter, 2), np.nan)

    # features & weights
    gdf.to_crs(crs_proj, inplace = True)
    feature_keys = ['x', 'y']
    gdf[feature_keys[0]] = gdf.geometry.x
    gdf[feature_keys[1]] = gdf.geometry.y
    gdf.to_crs(crs, inplace = True)

    # iterative clustering
    gdf[cluster_key] = pd.NA
    gdf[cluster_key] = gdf[cluster_key].astype('string')
    for i in tqdm(range(n_iter)):

        # fit model (unclustered only)
        clus_msk = gdf[cluster_key].notna()
        gdf_other = gdf.loc[~clus_msk].copy()
        gdf_other = fit_density_model(gdf = gdf_other,
                                      eps_dbscan = eps_dbscan,
                                      min_samples_dbscan = int(min_samples_dbscan),
                                      feature_keys = feature_keys,
                                      cluster_key = cluster_key,
                                      population_key = population_key,
                                      name_keys = name_keys)
        gdf.update(gdf_other)

        # compute distance per capita
        if return_metrics or ((i == n_iter - 1) and return_gdf):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category = FutureWarning)
                gdf_dpc = evaluate_density(gdf = gdf,
                                           cluster_key = cluster_key,
                                           population_key = population_key,
                                           crs_proj = crs_proj)

        # log metrics
        if return_metrics:
            clus_msk = gdf_dpc[cluster_key].notna()
            metrics[i, ix_clus] = gdf_dpc.loc[clus_msk, metric_key].mean()
            metrics[i, ix_other] = gdf_dpc.loc[~clus_msk, metric_key].mean()

        # decrease min population
        min_samples_dbscan *= min_samples_decay

    # clean up
    if return_gdf:
        gdf_dpc.drop(columns = feature_keys, inplace = True)

    results = ()
    if return_metrics:
        results += (metrics,)
    if return_gdf:
        results += (gdf_dpc,)

    if len(results) == 1:
        return results[0]
    elif results:
        return results
    else:
        return None
    

def fit_density_model(gdf: gpd.GeoDataFrame,
                      eps_dbscan: float,
                      min_samples_dbscan: int,
                      feature_keys: list[str] = ['x', 'y'],
                      cluster_key: str = 'cluster',
                      population_key: str = 'population',
                      name_keys: str | list[str] = ['city', 'state', 'country']
                      ) -> gpd.GeoDataFrame:

    # fit model
    gdf = gdf.copy()
    X = gdf[feature_keys].values
    w = gdf[population_key].values
    model = DBSCAN(eps = eps_dbscan, min_samples = min_samples_dbscan)
    gdf[cluster_key] = model.fit_predict(X, sample_weight = w)

    # remove singleton clusters & rename
    n_obs = gdf[cluster_key].value_counts()
    obs_msk = gdf[cluster_key].map(n_obs) > 1
    gdf[cluster_key] = gdf[cluster_key].where(obs_msk, -1)
    gdf = rename_clusters(gdf, cluster_key, population_key, name_keys)
    return gdf


def rename_clusters(gdf: gpd.GeoDataFrame,
                    cluster_key: str,
                    population_key: str,
                    name_keys: str | list[str]
                    ) -> gpd.GeoDataFrame:
    
    if isinstance(name_keys, str):
        name_keys = [name_keys]
    
    gdf = gdf.copy()
    clus_msk = gdf[cluster_key] >= 0    # -1 = unassigned
    clusname_map = (gdf.loc[clus_msk]
                    .sort_values(population_key, ascending = False)
                    .drop_duplicates(cluster_key)
                    .set_index(cluster_key)
                    .apply(lambda row: ' '.join(str(row[col]) for col in name_keys), axis = 1))
    gdf['cluster_name'] = gdf[cluster_key].map(clusname_map)
    gdf.loc[~clus_msk, 'cluster_name'] = pd.NA
    gdf[cluster_key] = gdf['cluster_name'].astype('string')
    gdf.drop('cluster_name', axis = 1, inplace = True)
    return gdf
