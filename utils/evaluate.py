import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.axes import Axes
from scipy.spatial.distance import jensenshannon

def get_dpc_clusters(gdf: gpd.GeoDataFrame,
                     cluster_key: str,
                     population_key: str = 'population',
                     crs_proj: str = 'EPSG:4087',
                     return_sorted: bool = True
                     ) -> gpd.GeoDataFrame:

    crs = gdf.crs
    gdf = gdf.copy()
    gdf.to_crs(crs_proj, inplace = True)

    # distance to cluster centroid (weighted)
    centroids_map = (gdf.groupby(cluster_key).apply(lambda row: Point(
                      np.average(row.geometry.x, weights = row[population_key]),
                      np.average(row.geometry.y, weights = row[population_key]))))
    centroids = gpd.GeoSeries(gdf[cluster_key].map(centroids_map), crs = crs_proj)
    gdf['d_centroid'] = gdf.geometry.distance(centroids)
    gdf[f'{cluster_key}_ref'] = gdf[cluster_key].copy()

    # total population (cluster)
    population = gdf.groupby(cluster_key)[population_key].sum()
    gdf['total_population'] = gdf[cluster_key].map(population)

    # distance per capita
    gdf['log10_dpc'] = np.log10(gdf.d_centroid / gdf.total_population)
    gdf.drop(columns = ['d_centroid', 'total_population'], inplace = True)

    if return_sorted:
        gdf.sort_values('log10_dpc', inplace = True)
    
    gdf.to_crs(crs, inplace = True)
    return gdf


def get_dpc_unassigned(gdf: gpd.GeoDataFrame,
                       gdf_ref: gpd.GeoDataFrame,
                       cluster_key: str,
                       population_key: str = 'population',
                       crs_proj: str = 'EPSG:4087',
                       return_sorted: bool = True
                       ) -> gpd.GeoDataFrame:

    crs = gdf.crs
    gdf = gdf.to_crs(crs_proj).copy()
    gdf_ref = gdf_ref.to_crs(crs_proj).copy()

    # centroids & populations (ref clusters)
    centroids = gdf_ref.groupby(cluster_key).apply(lambda row: Point(
                    np.average(row.geometry.x, weights = row[population_key]),
                    np.average(row.geometry.y, weights = row[population_key])))
    gdf_centroids = gpd.GeoDataFrame(
        {'cluster_ref': centroids.index, 'geometry': centroids.values},
        geometry = 'geometry', crs = crs_proj)
    population = gdf_ref.groupby(cluster_key)[population_key].sum()
    gdf_centroids['total_population'] = gdf_centroids['cluster_ref'].map(population)

    # nearest centroids to unassigned
    gdf_return = gpd.sjoin_nearest(
        gdf, gdf_centroids,
        how = 'left',
        lsuffix = 'orig',
        rsuffix = 'ref',
        distance_col = 'd_centroid')
    gdf_return.rename(columns = {'cluster_ref' : f'{cluster_key}_ref'}, inplace = True)
    gdf_return.drop('index_ref', axis = 1, inplace = True)

    # distance per capita
    gdf_return['log10_dpc'] = np.log10(gdf_return.d_centroid / gdf_return.total_population)
    gdf_return.drop(columns = ['d_centroid', 'total_population'], inplace = True)
    
    if return_sorted:
        gdf_return.sort_values('log10_dpc', inplace = True)

    gdf_return.to_crs(crs, inplace = True)
    return gdf_return


def get_jsdiv(gdf1: gpd.GeoDataFrame,
              gdf2: gpd.GeoDataFrame,
              metric_key: str = 'log10_dpc',
              n_bins: int = 100,
              eps: float = 1e-6,
              plot: bool = False,
              ax: Axes | None = None,
              return_ax: bool = False,
              gdf1_label: str | None = None,
              gdf2_label: str | None = None,
              gdf1_facecolor: str = 'cornflowerblue',
              gdf2_facecolor: str = 'tomato',
              facecolor_alpha: float = .33
              ) -> float | tuple[float, Axes]:
    
    # define metric range
    metric1 = gdf1[metric_key]
    metric2 = gdf2[metric_key]
    metric_min = min(metric1.min(), metric2.min())
    metric_max = max(metric1.max(), metric2.max())
    bins = np.linspace(metric_min, metric_max, n_bins)
    
    # compute probability densities
    p_hist, _ = np.histogram(metric1, bins = bins, density = True)
    q_hist, _ = np.histogram(metric2, bins = bins, density = True)
    p_hist += eps
    q_hist += eps
    p = p_hist / p_hist.sum()
    q = q_hist / q_hist.sum()

    # Jensen-Shannon divergence
    jsdiv = jensenshannon(p, q, base = 2)

    if plot:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize = (6, 2.5))

        ax.hist(metric1,
                bins = bins,
                facecolor = gdf1_facecolor,
                alpha = facecolor_alpha,
                density = True,
                label = gdf1_label)
        
        ax.hist(metric2,
                bins = bins,
                facecolor = gdf2_facecolor,
                alpha = facecolor_alpha,
                density = True,
                label = gdf2_label)
        
        if return_ax:
            return jsdiv, ax
        else:
            return jsdiv
    else:
        return jsdiv
