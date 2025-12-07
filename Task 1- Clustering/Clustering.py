import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from Clustering_Plots import save_metrics_to_pdf,create_dataframe

# --- 1. K-Means ---
def run_kmeans_analysis(df_scaled, base_path, k_range=range(2, 11)):
    folderpath = os.path.join(base_path, "KMeans")
    if not os.path.exists(folderpath): os.makedirs(folderpath)
    
    metrics = []
    print("Running K-Means...")
    
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = model.fit_predict(df_scaled)
        
        metrics.append({
            'K': k,
            'Inertia': model.inertia_,
            'Silhouette': silhouette_score(df_scaled, labels),
            'Davies-Bouldin': davies_bouldin_score(df_scaled, labels),
            'Calinski-Harabasz': calinski_harabasz_score(df_scaled, labels)
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    # Save results
    metrics_df.to_markdown(os.path.join(folderpath, 'kmeans_table.md'), index=False)
    save_metrics_to_pdf(metrics_df, os.path.join(folderpath, 'kmeans_plots.pdf'), "K-Means Evaluation")
    return metrics_df

# --- 2. Spectral Clustering ---
def run_spectral_analysis(df_scaled, base_path, k_range=range(2, 11)):
    folderpath = os.path.join(base_path, "Spectral")
    if not os.path.exists(folderpath): os.makedirs(folderpath)
    
    metrics = []
    print("Running Spectral Clustering (this may be slow)...")
    
    for k in k_range:
        # eigen_solver='arpack' is generally more stable for larger matrices
        model = SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors', n_neighbors=10)
        labels = model.fit_predict(df_scaled)
        
        metrics.append({
            'K': k,
            # Inertia is not defined for Spectral Clustering
            'Silhouette': silhouette_score(df_scaled, labels),
            'Davies-Bouldin': davies_bouldin_score(df_scaled, labels),
            'Calinski-Harabasz': calinski_harabasz_score(df_scaled, labels)
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    metrics_df.to_markdown(os.path.join(folderpath, 'spectral_table.md'), index=False)
    save_metrics_to_pdf(metrics_df, os.path.join(folderpath, 'spectral_plots.pdf'), "Spectral Clustering Evaluation")
    return metrics_df

# --- 3. Gaussian Mixture Models (GMM) ---
def run_gmm_analysis(df_scaled, base_path, k_range=range(2, 11)):
    folderpath = os.path.join(base_path, "GMM")
    if not os.path.exists(folderpath): os.makedirs(folderpath)
    
    metrics = []
    print("Running Gaussian Mixture Models...")
    
    for k in k_range:
        model = GaussianMixture(n_components=k, random_state=42)
        labels = model.fit_predict(df_scaled)
        
        metrics.append({
            'K': k,
            'AIC': model.aic(df_scaled), # Specific to GMM (Lower is better)
            'BIC': model.bic(df_scaled), # Specific to GMM (Lower is better)
            'Silhouette': silhouette_score(df_scaled, labels),
            'Davies-Bouldin': davies_bouldin_score(df_scaled, labels)
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    metrics_df.to_markdown(os.path.join(folderpath, 'gmm_table.md'), index=False)
    save_metrics_to_pdf(metrics_df, os.path.join(folderpath, 'gmm_plots.pdf'), "GMM Evaluation (AIC/BIC)")
    return metrics_df

# --- 4. DBSCAN (Independent Research) ---
def run_dbscan_analysis(df_scaled, base_path, eps_range=np.arange(0.1, 3.0, 0.2)):
    """
    DBSCAN iterates over 'eps' (distance), not K. 
    Metrics must filter out Noise (-1) labels to be valid.
    """
    folderpath = os.path.join(base_path, "DBSCAN")
    if not os.path.exists(folderpath): os.makedirs(folderpath)
    
    metrics = []
    print("Running DBSCAN Analysis...")
    
    for eps in eps_range:
        # min_samples usually = 2*dimensions or just 5 for general 2D/3D work.
        # Given we have 18 dims (or 2 if PCA), let's stick to default 5 for now.
        model = DBSCAN(eps=eps, min_samples=5)
        labels = model.fit_predict(df_scaled)
        
        # Count clusters (excluding noise -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Calculate silhouette ONLY if we found > 1 cluster and they aren't all noise
        if n_clusters > 1:
            # Mask noise points for metric calculation
            # We want to know how good the ACTUAL clusters are
            core_samples_mask = labels != -1
            if sum(core_samples_mask) > 0:
                sil = silhouette_score(df_scaled[core_samples_mask], labels[core_samples_mask])
            else:
                sil = -1 # All noise
        else:
            sil = -1 # Not enough clusters to evaluate
            
        metrics.append({
            'Eps': round(eps, 2),
            'Clusters_Found': n_clusters,
            'Noise_Points': n_noise,
            'Silhouette (No Noise)': sil
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    metrics_df.to_markdown(os.path.join(folderpath, 'dbscan_table.md'), index=False)
    # Use the universal plotter (it will detect 'Eps' as x-axis)
    save_metrics_to_pdf(metrics_df, os.path.join(folderpath, 'dbscan_plots.pdf'), "DBSCAN Evaluation")
    return metrics_df


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Ensure you import your universal plotter
# from visualization import save_metrics_to_pdf 

# --- 5. Agglomerative Hierarchical Clustering (Bottom-Up) ---
def run_agglomerative_analysis(df_scaled, base_path, k_range=range(2, 11)):
    folderpath = os.path.join(base_path, "Agglomerative")
    if not os.path.exists(folderpath): os.makedirs(folderpath)
    
    metrics = []
    print("Running Agglomerative Clustering...")
    
    for k in k_range:
        # 'ward' linkage minimizes variance (like K-Means) and is robust
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(df_scaled)
        
        metrics.append({
            'K': k,
            # Inertia is not calculated in standard Agglomerative Clustering
            'Silhouette': silhouette_score(df_scaled, labels),
            'Davies-Bouldin': davies_bouldin_score(df_scaled, labels),
            'Calinski-Harabasz': calinski_harabasz_score(df_scaled, labels)
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    # Save Results
    metrics_df.to_markdown(os.path.join(folderpath, 'agglomerative_table.md'), index=False)
    
    # Assuming save_metrics_to_pdf is defined in your visualization file
    save_metrics_to_pdf(metrics_df, os.path.join(folderpath, 'agglomerative_plots.pdf'), "Agglomerative Evaluation")
    
    # --- BONUS: Dendrogram Plot ---
    # Top marks require visualizing the hierarchy, not just the cuts (K)
    plt.figure(figsize=(12, 6))
    plt.title("Agglomerative Dendrogram (Ward Linkage)")
    
    # Compute linkage matrix just for plotting (scipy logic)
    linkage_matrix = linkage(df_scaled, method='ward')
    dendrogram(linkage_matrix, truncate_mode='level', p=5) # p=5 shows only top 5 levels for clarity
    
    plt.xlabel("Number of points in node (or index of point if no parenthesis)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(folderpath, 'dendrogram.pdf'))
    plt.close()
    print(f"Saved Dendrogram to {folderpath}")
    
    return metrics_df

# --- 6. Divisive Hierarchical Clustering (Top-Down via Bisecting K-Means) ---
def run_divisive_analysis(df_scaled, base_path, k_range=range(2, 11)):
    """
    Implements Divisive clustering using Bisecting K-Means.
    This starts with 1 cluster and recursively splits it.
    """
    folderpath = os.path.join(base_path, "Divisive_BisectingKMeans")
    if not os.path.exists(folderpath): os.makedirs(folderpath)
    
    metrics = []
    print("Running Divisive Analysis (Bisecting K-Means)...")
    
    for k in k_range:
        # BisectingKMeans was added in sklearn 1.1
        model = BisectingKMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(df_scaled)
        
        metrics.append({
            'K': k,
            'Inertia': model.inertia_, # Bisecting K-Means HAS inertia
            'Silhouette': silhouette_score(df_scaled, labels),
            'Davies-Bouldin': davies_bouldin_score(df_scaled, labels),
            'Calinski-Harabasz': calinski_harabasz_score(df_scaled, labels)
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    metrics_df.to_markdown(os.path.join(folderpath, 'divisive_table.md'), index=False)
    save_metrics_to_pdf(metrics_df, os.path.join(folderpath, 'divisive_plots.pdf'), "Divisive (Bisecting K-Means) Evaluation")
    
    return metrics_df

if __name__ == "__main__":
    # base_path = os.path.join(os.getcwd(),"No_Preprocessing_Cluster_Analysis")
    # filepath = os.path.join(os.getcwd(),'Data Files','ClimateDataBasel.csv')
    # main_df = create_dataframe(filepath=filepath)

    # #flow 1 where we run clustering without any pre-processing
    # run_kmeans_analysis(main_df,base_path=base_path)
    # run_dbscan_analysis(main_df,base_path=base_path)
    # run_gmm_analysis(main_df,base_path=base_path)
    # run_spectral_analysis(main_df,base_path=base_path)
    # run_divisive_analysis(main_df,base_path=base_path)
    # run_agglomerative_analysis(main_df,base_path=base_path)

    # #flow 2 where we run clustering after scaling of columns 
    # base_path = os.path.join(os.getcwd(),"Scaled_Cluster_Analysis")
    # filepath = os.path.join(os.getcwd(),'Data Files','Climate_Data_Scaled.csv')
    # main_df = pd.read_csv(filepath)
    # #flow 1 where we run clustering without any pre-processing
    # run_kmeans_analysis(main_df,base_path=base_path)
    # run_dbscan_analysis(main_df,base_path=base_path)
    # run_gmm_analysis(main_df,base_path=base_path)
    # run_spectral_analysis(main_df,base_path=base_path)
    # run_divisive_analysis(main_df,base_path=base_path)
    # run_agglomerative_analysis(main_df,base_path=base_path)


    #flow 3 where we run it with scaled and removed columns
    base_path = os.path.join(os.getcwd(),"Scaled_Removed_Cluster_Analysis")
    filepath = os.path.join(os.getcwd(),'Data Files','Climate_Data_Scaled_Removed.csv')
    main_df = pd.read_csv(filepath)
    run_kmeans_analysis(main_df,base_path=base_path)
    run_dbscan_analysis(main_df,base_path=base_path)
    run_gmm_analysis(main_df,base_path=base_path)
    run_spectral_analysis(main_df,base_path=base_path)
    run_divisive_analysis(main_df,base_path=base_path)
    run_agglomerative_analysis(main_df,base_path=base_path)

    #flow 4 where we run with check PCA scaled and removed
    base_path = os.path.join(os.getcwd(),"PCA_Scaled_Removed_Cluster_Analysis")
    filepath = os.path.join(os.getcwd(),'Data Files','Climate_Data_Scaled_Removed_PCA.csv')
    main_df = pd.read_csv(filepath)
    run_kmeans_analysis(main_df,base_path=base_path)
    run_dbscan_analysis(main_df,base_path=base_path)
    run_gmm_analysis(main_df,base_path=base_path)
    run_spectral_analysis(main_df,base_path=base_path)
    run_divisive_analysis(main_df,base_path=base_path)
    run_agglomerative_analysis(main_df,base_path=base_path)