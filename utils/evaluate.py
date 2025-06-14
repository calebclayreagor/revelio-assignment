import numpy as np
import geopandas as gpd

def evaluate_clusters(gdf: gpd.GeoDataFrame,
                      cluster_key: str,
                      population_key: str = 'population',
                      crs_proj: str = 'EPSG:4087'):

    crs = gdf.crs
    gdf.to_crs(crs_proj, inplace = True)

    # distance to cluster centroid
    centroids = gdf.dissolve(cluster_key).centroid
    gdf[f'{cluster_key}_centroid'] = gdf[cluster_key].map(centroids)
    gdf['d_centroid'] = gdf.distance(gdf[f'{cluster_key}_centroid'])

    # total cluster population
    population = gdf.groupby(cluster_key)[population_key].sum()
    gdf[f'{cluster_key}_{population_key}'] = gdf[cluster_key].map(population)

    # distance per capita (dpc)
    gdf['dpc'] = gdf.d_centroid / gdf[f'{cluster_key}_{population_key}']
    gdf['log_dpc'] = np.log10(gdf.dpc)
    gdf.sort_values('dpc', inplace = True)
    gdf.to_crs(crs, inplace = True)


def evaluate_unassigned(gdf: gpd.GeoDataFrame,
                        gdf_ref: gpd.GeoDataFrame,
                        cluster_key: str,
                        population_key: str = 'population',
                        crs_proj: str = 'EPSG:4087'
                        ) -> gpd.GeoDataFrame:

    crs = gdf.crs
    gdf.to_crs(crs_proj, inplace = True)
    gdf_ref_copy = gdf_ref.to_crs(crs_proj).copy()

    # ref clusters — centroids & populations
    centroids = gdf_ref_copy.dissolve(cluster_key).centroid
    gdf_centroids = gpd.GeoDataFrame(
        {f'{cluster_key}_ref' : centroids.index, 'geometry' : centroids},
        geometry = 'geometry', crs = crs_proj)
    population = gdf_ref_copy.groupby(cluster_key)[population_key].sum()
    gdf_centroids[f'{cluster_key}_ref_{population_key}'] = gdf_centroids[f'{cluster_key}_ref'].map(population)
    del gdf_centroids[f'{cluster_key}_ref']

    # unassigned — nearest centroids
    gdf = gpd.sjoin_nearest(
        gdf, gdf_centroids,
        how = 'left',
        lsuffix = 'orig',
        rsuffix = 'ref',
        distance_col = 'd_centroid')

    # distance per capita (dpc)
    gdf['dpc'] = gdf.d_centroid / gdf[f'{cluster_key}_ref_{population_key}']
    gdf['log_dpc'] = np.log10(gdf.dpc)
    gdf.sort_values('dpc', inplace = True)
    gdf.to_crs(crs, inplace = True)
    return gdf
