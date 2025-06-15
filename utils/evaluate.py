import numpy as np
import geopandas as gpd
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def get_dpc_clusters(gdf: gpd.GeoDataFrame,
                     cluster_key: str,
                     population_key: str = 'population',
                     crs_proj: str = 'EPSG:4087',
                     return_log: bool = True,
                     return_sorted: bool = True,
                     inplace: bool = True
                     ) -> None | gpd.GeoDataFrame:

    if not inplace:
        gdf = gdf.copy()

    crs = gdf.crs
    gdf.to_crs(crs_proj, inplace = True)

    # distance to cluster centroid
    centroids = gdf.dissolve(cluster_key).centroid
    gdf[f'{cluster_key}_centroid'] = gdf[cluster_key].map(centroids)
    gdf['d_centroid'] = gdf.distance(gdf[f'{cluster_key}_centroid'])
    gdf.drop(f'{cluster_key}_centroid', axis = 1, inplace = True)

    # cluster total population
    population = gdf.groupby(cluster_key)[population_key].sum()
    gdf[f'{cluster_key}_{population_key}'] = gdf[cluster_key].map(population)

    # distance per capita
    gdf['dpc'] = gdf.d_centroid / gdf[f'{cluster_key}_{population_key}']
    gdf.drop(columns = ['d_centroid', f'{cluster_key}_{population_key}'], inplace = True)

    if return_log:
        gdf['log10_dpc'] = np.log10(gdf.dpc)

    if return_sorted:
        gdf.sort_values('dpc', inplace = True)
    
    gdf.to_crs(crs, inplace = True)

    if not inplace:
        return gdf


def get_dpc_unassigned(gdf: gpd.GeoDataFrame,
                       gdf_ref: gpd.GeoDataFrame,
                       cluster_key: str,
                       population_key: str = 'population',
                       crs_proj: str = 'EPSG:4087',
                       return_log: bool = True,
                       return_sorted: bool = True
                       ) -> gpd.GeoDataFrame:

    crs = gdf.crs
    gdf = gdf.to_crs(crs_proj).copy()
    gdf_ref = gdf_ref.to_crs(crs_proj).copy()

    # ref clusters — centroids & populations
    centroids = gdf_ref.dissolve(cluster_key).centroid
    gdf_centroids = gpd.GeoDataFrame(
        {f'{cluster_key}_ref' : centroids.index, 'geometry' : centroids},
        geometry = 'geometry', crs = crs_proj)
    population = gdf_ref.groupby(cluster_key)[population_key].sum()
    gdf_centroids[f'{cluster_key}_ref_{population_key}'] = gdf_centroids[f'{cluster_key}_ref'].map(population)
    gdf_centroids.drop(f'{cluster_key}_ref', axis = 1, inplace = True)

    # unassigned — nearest centroids
    gdf_return = gpd.sjoin_nearest(
        gdf, gdf_centroids,
        how = 'left',
        lsuffix = 'orig',
        rsuffix = 'ref',
        distance_col = 'd_centroid')
    gdf_return[cluster_key] = gdf_return[f'{cluster_key}_orig']
    gdf_return.drop(f'{cluster_key}_orig', axis = 1, inplace = True)

    # distance per capita
    gdf_return['dpc'] = gdf_return.d_centroid / gdf_return[f'{cluster_key}_ref_{population_key}']
    gdf_return.drop(columns = ['d_centroid', f'{cluster_key}_ref_{population_key}'], axis = 1, inplace = True)

    if return_log:
        gdf_return['log10_dpc'] = np.log10(gdf_return.dpc)
    
    if return_sorted:
        gdf_return.sort_values('dpc', inplace = True)

    gdf_return.to_crs(crs, inplace = True)
    return gdf_return


def get_jsdiv(gdf1: gpd.GeoDataFrame,
              gdf2: gpd.GeoDataFrame,
              metric_key: str,
              n_bins: int = 100,
              eps: float = 1e-3,
              plot: bool = False,
              ax: Axes | None = None,
              return_ax: bool = False,
              gdf1_label: str | None = None,
              gdf2_label: str | None = None,
              gdf1_facecolor: str | None = None,
              gdf2_facecolor: str | None = None,
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
        if gdf1_label is None:
            gdf1_label = 'GeoDataFrame1'

        if gdf2_label is None:
            gdf2_label = 'GeoDataFrame2'

        if gdf1_facecolor is None:
            gdf1_facecolor = 'cornflowerblue'

        if gdf2_facecolor is None:
            gdf2_facecolor = 'tomato'

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
