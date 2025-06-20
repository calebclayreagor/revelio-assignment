# Iterative density-based clustering of metropolitan areas

## Motivation

To define a metropolitan area, one might rely on geography, history, or culture, but an accurate assessment of urban centers likely requires an objective measure that can capture the contributions of multiple factors. Because traditional definitions of metropolitan statistical areas (MSAs) rely on hand-drawn boundaries and reflect subtle biases, it is necessary to develop a data-driven approach to refine or validate these designations. A principled approach might also uncover overlooked regions with substantial growth potential, both in the US and internationally.

## Approach

### Distance per capita

To provide an unbiased assessment of traditional and algorithmically defined urban clusters, I developed a novel metric called distance per capita (DPC) that measures a city's proximity to its nearest metropolitan area, relative to that area's aggregate population:

$DPC=\frac{\text{distance to the nearest cluster centroid}}{\text{total population of the nearest cluster}} \propto \text{density}^{-1}$

where each centroid is defined as the population-weighted average location of cities belonging to that cluster. For cities in metropolitan areas, the nearest centroids correspond to their own clusters, whereas nonmetropolitan areas are evaluated relative to their closest non-member clusters. Because DPC assesses a city's proximity relative to an area's total population, this metric is scale-invariant and can describe clusters of both large and small cities.

My function [`utils.evaluate.evaluate_density`](utils/evaluate.py#L9-42) computes the $log_{10}$ $\text{DPC}$ for any arbitrary clustering scheme.

### Urban *versus* rural separation

Building on the core concept of DPC, I devised an estimate for the quality of nonmetropolitan-area identification based on statistical entropy. For a given geographical area and clustering scheme, I defined the separation of metropolitan *versus* nonmetropolitan areas as the Jensen-Shannon divergence between their DPC distributions:

$JS_{div}=\sqrt{\frac{KL_{k,\varnothing} + KL_{\varnothing,k}}{2}} \in [0, 1]$

where $KL_{i,j}$ is the Kullback-Leibler divergence between distributions $i,j$, and $k,\varnothing$ represent the $log_{10}$ $\text{DPC}$ distributions for the clustered and unassigned cities, respectively. Values of $JS_{div}$ near 0 indicate poor separation, whereas values near 1 indicate significant heterogeneity between urban (clustered) and rural (unassigned) areas.

My function [`utils.evaluate.get_jsdiv`](utils/evaluate.py#L122-190) computes the $JS_{div}$ between clustered and unassigned cities.

### Iterative density-based clustering

Because the data contained relatively few features except for cities' locations and populations, I employed a density-based clustering approach with suitable bias-variance tradeoff. In particular, I used the DBSCAN algorithm (<ins>D</ins>ensity-<ins>B</ins>ased <ins>S</ins>patial <ins>C</ins>lustering of <ins>A</ins>pplications with <ins>N</ins>oise), which performs partial clustering based on sample-weighted spatial density. However, DBSCAN tends to struggle on heterogeneous and multi-scale data due to its reliance on a fixed neighborhood radius $\epsilon$; hierarchical variants such as <ins>H</ins>DBSCAN can address this limitation, but current implementations neglect sample weighting. To cluster cities and define meaningful metropolitan areas, I developed the following iterative density-based clustering approach using population sizes as sample weights:

> 1. Initialize all cities $i$ as unclustered $\varnothing$
>
> 2. Initialize the minimum population-size threshold $w_{min}$
>
> 3. Repeat for $N$ iterations:
> 
>    a. Select all unclustered cities $i_{\varnothing}$
>
>    b. Apply DBSCAN using coordinates $x_i,y_i$ and population sizes $w_i$
>
>    c. Assign labeled cities $i$ to newly-found clusters $k$
>
>    f. Decay the minimum population threshold by a factor $\lambda$
>
> 4. Return final cluster assignments $i_{k,\varnothing}$ and metrics $\text{DPC}_i$

This approach incrementally assigns cities to metropolitan areas from denser to sparser regions through gradual relaxation of the minimum sample-weight threshold in DBSCAN. For the final metropolitan-area assignments, I set the maximum number of iterations $N$ until convergence of the mean $\text{DPC}_{k,\varnothing}$ values for clustered and unassigned cities. 

My algorithm can be run using the function [`utils.modeling.fit_iterative_density_model`](utils/modeling.py#L9-87). The notebook [`density_based_modeling.ipynb`](notebooks/density_based_modeling.ipynb) fits my model to the provided data.

## Key Findings

The following findings are based on my analyses in [`msa_us_only.ipynb`](notebooks/evaluation/msa_us_only.ipynb), [`density_clusters_us_only.ipynb`](notebooks/evaluation/density_clusters_us_only.ipynb), and [`density_clusters_international.ipynb`](notebooks/evaluation/density_clusters_international.ipynb).

My final clustering results can be found in [`ClusteringResults.csv`](data/ClusteringResults.csv).

### Summary

| Evaluation metric                         | Traditional MSA | Density Clusters (US) | Density Clusters (International) |
| ----------------------------------------- | --------------- | --------------------- | -------------------------------- |
| Number of clusters                        | 336             | 699                   | 1,714                            |
| Mean cluster population                   | 574,443         | 300,466               | 779,703                          |
| Mean cities per cluster                   | 14.68           | 8.44                  | 7.47                             |
| Percent clustered (cities)                | 69%             | 82%                   | 76%                              |
| Percent clustered (population)            | 87%             | 94%                   | 56%                              |
| Mean $log_{10}$ $\text{DPC}$ (clustered)  | -1.07           | -1.52                 | -1.7                             |
| Mean $log_{10}$ $\text{DPC}$ (unassigned) | -0.49           | 0.04                  | -0.36                            |
| $JS_{div}$ (clustered vs. unassigned)     | 0.38            | 0.73                  | 0.63                             |

### Highlights

1. My algorithm identifies **more metropolitan areas** than the traditional approach (699 vs. 336) for US cities

    a. On average, my clusters have **fewer cities** (~8 vs. 15) **and people** (~300 vs. 575k) than traditional MSAs

    b. This is *not* due to the splitting of large cities (see [MSA](notebooks/evaluation/msa_us_only.ipynb#population-percentiles)/[density cluster](notebooks/evaluation/density_clusters_us_only.ipynb#population-percentiles) percentile distributions & [large-city Sankey plot](notebooks/evaluation/density_clusters_us_only.ipynb#sankey-plot-label-comparison))

    - In several cases, my algorithm actually *merges* adjacent large cities (*e.g.*, LA/Riverside, SF/San Jose, and DC/Baltimore)

    c. More clusters are therefore the result of **_increased resolution of smaller metropolitan areas_** compared to the traditional MSAs

    - Consequently, my algorithm clusters a greater proportion of total cities (82 vs. 69%) and people (94 vs. 87%) than traditional MSAs

        - These areas are **properly categorized as metropolitan** based on DPC convergence and $\sim 2 \times$ increase in $JS_{div}$ compared to traditional MSAs

2. My algorithm performs **slightly worse on international data** based on $JS_{div}$ (0.63 vs. 0.73 for US)

    a. International clusters are **extremely dense** compared to US clusters, with $> 2.5 \times$ mean population (~780 vs. 300k) despite comparable city numbers

    b. Fewer international cities belong to metropolitan areas than in the US (76 vs. 82%), comprising only 56% of the international population

    - Gradual population decreases for smaller clusters (see [percentile distribution](notebooks/evaluation/density_clusters_international.ipynb#population-percentiles)) suggest that **more international cities are truly nonmetropolitan**

    c. Inspection of [edge cases](notebooks/evaluation/density_clusters_international.ipynb#worst-best-cluster-assignments) with high DPC values suggests that European cities are the least suited to density-based clustering and analysis

## Limitations

When interpreting the results of my clustering and analysis, it is important to keep in mind several key limitations as well as potential steps for mitigation:

1. My algorithm currently uses a **fixed neighborhood radius $\epsilon$** of 30 km, which restricts the relative scale of identified clusters

    - Future fix — implement $\epsilon$ decay alongside population-threshold decay during iterations

2. Density-based clustering fails to identify meaningful patterns that define metropolitan areas in areas with uniform density (*e.g.*, Europe)

    - Future fix — fit a European-specific model with tuned hyperparameters that better separate uniformly-dense regions

3. Density-based clustering relies solely on spatial and population features, which limits its overall effectiveness

    - Future fix — use density-based clusters as starting point to training more sophisticated models on new input features
