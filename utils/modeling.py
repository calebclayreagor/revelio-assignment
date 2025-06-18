import geopandas as gpd
from sklearn.cluster import DBSCAN

def fit_density_model(gdf: gpd.GeoDataFrame,
                      eps_dbscan: float,
                      min_samples_dbscan: int,
                      feature_keys: list[str] = ['x', 'y'],
                      weight_key: str = 'w',
                      population_key = 'population',
                      min_population: float = 1e5,
                      min_size: int = 2,
                      ) -> gpd.GeoDataFrame:

    # fit model
    gdf = gdf.copy()
    X = gdf[feature_keys].values
    w = gdf[weight_key].values
    model = DBSCAN(eps = eps_dbscan, min_samples = min_samples_dbscan)
    gdf['cluster'] = model.fit_predict(X, sample_weight = w)

    # prune small clusters
    n_obs = gdf.cluster.value_counts()
    obs_msk = gdf.cluster.map(n_obs) >= min_size
    gdf['cluster'] = gdf.cluster.where(obs_msk, -1)

    # population threshold
    population = gdf.groupby('cluster')[population_key].sum()
    population_msk = gdf.cluster.map(population) >= min_population
    gdf['cluster'] = gdf.cluster.where(population_msk, -1)
    return gdf
