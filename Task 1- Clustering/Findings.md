Correlation -> some features highly correlated, might drop especially the ones with the same name.
Histogram -> most have a decent distribution except for snowfall amoount
BoxPlot ->


##  references
1. https://www.researchgate.net/profile/Trupti-Kodinariya/publication/313554124_Review_on_Determining_of_Cluster_in_K-means_Clustering/links/5789fda408ae59aa667931d2/Review-on-Determining-of-Cluster-in-K-means-Clustering.pdf (finding best K for clustering)



# things to do, read on how different metrics are scored
# read on different clustering algos and their metrics
# pre-processing



filepath = os.path.join(os.getcwd(),'Data Files','ClimateDataBasel.csv')


columns to keep{
temp(mean),
rel_humid(mean),
precipitation_total,
sunshine_duration,
snowfall_amount,
sea_level_pressue(mean),
wind_gust(mean),
wind_speed(mean)
}