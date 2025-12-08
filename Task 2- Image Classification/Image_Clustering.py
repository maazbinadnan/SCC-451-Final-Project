import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

class ClusteringAnalysis:
    def __init__(self, feature_df: pd.DataFrame):
        """
        Args:
            feature_df: DataFrame containing feature vectors. 
                        Assumes 'Label' column exists and rest are features.
        """
        if 'Label' not in feature_df.columns:
            raise ValueError("DataFrame must have a 'Label' column for evaluation.")
            
        self.labels_true = feature_df['Label']
        self.X = feature_df.drop(columns=['Label']).values
        
        # Standardize features before clustering (Critical for K-Means)
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        
        print(f"Initialized Clustering with {self.X.shape[0]} samples and {self.X.shape[1]} dimensions.")

    def plot_elbow_method(self, max_k=10, save_path=None):
        """
        Runs K-Means for k=1 to max_k and plots the Inertia (Sum of Squared Errors)
        to find the 'Elbow'.
        """
        print("Calculating Elbow Method...")
        inertia = []
        K_range = range(1, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            inertia.append(kmeans.inertia_)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertia, 'bx-')
        plt.xlabel('k (Number of Clusters)')
        plt.ylabel('Inertia (Sum of Squared Distances)')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        
        if save_path:
            plt.savefig(f"{save_path}/elbow_plot.png")
            print(f"Elbow plot saved to {save_path}")
        
        plt.show()

    def run_clustering(self, algorithm='kmeans', n_clusters=2):
        """
        Runs the specified clustering algorithm and prints metrics.
        Args:
            algorithm: 'kmeans' or 'agglomerative'
        """
        print(f"\n--- Running {algorithm.upper()} (k={n_clusters}) ---")

    def plot_clustering_metrics(self, max_k=10, save_path=None):
        """
        Runs K-Means for k=2 to max_k (Silhouette requires at least 2 clusters)
        and plots Silhouette Score and Davies-Bouldin Index.
        """
        print(f"Calculating Metrics for k=2 to {max_k}...")
        
        silhouette_scores = []
        db_scores = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            # Re-fit the model for this specific k
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            
            # Calculate metrics
            sil = silhouette_score(self.X_scaled, labels)
            db = davies_bouldin_score(self.X_scaled, labels)
            
            silhouette_scores.append(sil)
            db_scores.append(db)

        # --- PLOTTING ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Silhouette Score (Higher is Better)
        ax1.plot(K_range, silhouette_scores, 'bo-', linewidth=2)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Analysis (Higher is Better)')
        ax1.grid(True)
        
        # Mark the best k
        best_k_sil = K_range[np.argmax(silhouette_scores)]
        ax1.axvline(best_k_sil, color='r', linestyle='--', label=f'Best k={best_k_sil}')
        ax1.legend()

        # Plot 2: Davies-Bouldin (Lower is Better)
        ax2.plot(K_range, db_scores, 'rs-', linewidth=2)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Davies-Bouldin Index')
        ax2.set_title('Davies-Bouldin Analysis (Lower is Better)')
        ax2.grid(True)
        
        # Mark the best k
        best_k_db = K_range[np.argmin(db_scores)]
        ax2.axvline(best_k_db, color='b', linestyle='--', label=f'Best k={best_k_db}')
        ax2.legend()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/metrics_evaluation.png")
            print(f"Metrics plot saved to {save_path}")
            
        plt.show()