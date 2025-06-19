import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.axes import Axes
from scipy.spatial.distance import jensenshannon

def evaluate_density(gdf: gpd.GeoDataFrame,
                     cluster_key: str,
                     population_key: str = 'population',
                     crs_proj: str = 'EPSG:4087',
                     return_sorted: bool = True
                     ) -> gpd.GeoDataFrame:
    """Compute log10 distance-per-capita (DPC) for clustered and unassigned cities"""
    
    # split clustered vs. unassigned
    gdf = gdf.copy()
    clus_msk = gdf[cluster_key].notna()
    gdf_clus = gdf.loc[clus_msk].copy()
    gdf_other = gdf.loc[~clus_msk].copy()

    # compute distance per capita
    gdf_clus = get_dpc_clusters(gdf = gdf_clus,
                                cluster_key = cluster_key,
                                population_key = population_key,
                                crs_proj = crs_proj,
                                return_sorted = return_sorted)
    gdf_other = get_dpc_unassigned(gdf = gdf_other,
                                   gdf_ref = gdf_clus,
                                   cluster_key = cluster_key,
                                   population_key = population_key,
                                   crs_proj = crs_proj,
                                   return_sorted = return_sorted)
    gdf_return = pd.concat((gdf_clus, gdf_other), axis = 0)

    if return_sorted:
        gdf_return.sort_values('log10_dpc', inplace = True)
    else:
        gdf_return = gdf_return.loc[gdf.index]

    return gdf_return


def get_dpc_clusters(gdf: gpd.GeoDataFrame,
                     cluster_key: str,
                     population_key: str = 'population',
                     crs_proj: str = 'EPSG:4087',
                     return_sorted: bool = True
                     ) -> gpd.GeoDataFrame:

    crs = gdf.crs
    gdf = gdf.copy()
    gdf.to_crs(crs_proj, inplace = True)

    # distance to population-weighted cluster centroid
    centroids_map = (gdf.groupby(cluster_key).apply(lambda row: Point(
                      np.average(row.geometry.x, weights = row[population_key]),
                      np.average(row.geometry.y, weights = row[population_key]))))
    centroids = gpd.GeoSeries(gdf[cluster_key].map(centroids_map), crs = crs_proj)
    gdf['d_centroid'] = gdf.geometry.distance(centroids)
    gdf[f'{cluster_key}_ref'] = gdf[cluster_key].copy()

    # total cluster population
    population = gdf.groupby(cluster_key)[population_key].sum()
    gdf['total_population'] = gdf[cluster_key].map(population)

    # distance per capita (log scale)
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

    # distance per capita (log scale)
    gdf_return['log10_dpc'] = np.log10(gdf_return.d_centroid / gdf_return.total_population)
    gdf_return.drop(columns = ['d_centroid', 'total_population'], inplace = True)
    
    if return_sorted:
        gdf_return.sort_values('log10_dpc', inplace = True)

    gdf_return.to_crs(crs, inplace = True)
    return gdf_return


def get_jsdiv(gdf: gpd.GeoDataFrame,
              cluster_key: str,
              metric_key: str = 'log10_dpc',
              n_bins: int = 100,
              eps: float = 1e-6,
              plot: bool = False,
              ax: Axes | None = None,
              return_ax: bool = False,
              clus_label: str | None = None,
              other_label: str | None = None,
              clus_facecolor: str = 'cornflowerblue',
              other_facecolor: str = 'tomato',
              facecolor_alpha: float = .33
              ) -> float | tuple[float, Axes]:
    """
    Compute Jensen-Shannon divergence between clustered and unassigned metric distributions.
    Optionally plot overlapping histograms.
    """
    
    # compute probability densities
    clus_msk = gdf[cluster_key].notna()
    metric_clus = gdf.loc[clus_msk, metric_key].values
    metric_other = gdf.loc[~clus_msk, metric_key].values
    bins = np.linspace(gdf[metric_key].min(), gdf[metric_key].max(), n_bins)
    p_hist, _ = np.histogram(metric_clus, bins = bins)
    q_hist, _ = np.histogram(metric_other, bins = bins)
    p_hist = p_hist.astype(np.float64)
    q_hist = q_hist.astype(np.float64)
    p_hist += eps
    q_hist += eps
    p = p_hist / p_hist.sum()
    q = q_hist / q_hist.sum()

    # Jensen-Shannon divergence
    jsdiv = jensenshannon(p, q, base = 2)

    if plot:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize = (6, 2.5))

        ax.hist(metric_clus,
                bins = bins,
                facecolor = clus_facecolor,
                alpha = facecolor_alpha,
                density = True,
                label = clus_label)
        
        ax.hist(metric_other,
                bins = bins,
                facecolor = other_facecolor,
                alpha = facecolor_alpha,
                density = True,
                label = other_label)
        
        if clus_label or other_label:
            ax.legend(frameon = False)
        
        if return_ax:
            return jsdiv, ax
        else:
            return jsdiv
    else:
        return jsdiv
