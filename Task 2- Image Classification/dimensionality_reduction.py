import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

class PCAHandler:
    def __init__(self, n_components=50):
        """
        Args:
            n_components: Number of components to keep. 
                          Use 2 for simple viz, or 50 for actual clustering/classification.
        """
        self.n_components = n_components
        self.pca = PCA()
        self.scaler = None
        self.feature_cols = 'Label'

    def fit_transform(self, dataframe: pd.DataFrame):
        """
        Fits PCA on the TRAINING set and returns the transformed data.
        """
        # 1. Identify Feature Columns (Exclude Label)
        if 'Label' not in dataframe.columns:
            raise ValueError("DataFrame must have a 'Label' column.")
        
        X = dataframe.drop(columns=self.feature_cols)
        y = dataframe['Label']

        # 2. Standardize (Fit & Transform)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 3. PCA (Fit & Transform)
        self.pca = PCA(n_components=self.n_components)
        principal_components = self.pca.fit_transform(X_scaled)

        # 4. Return Result as DataFrame
        col_names = [f'PC{i+1}' for i in range(self.n_components)]
        result_df = pd.DataFrame(data=principal_components, columns=col_names)
        result_df['Label'] = y.values
        
        return result_df

    def transform(self, dataframe: pd.DataFrame):
        """
        Applies the EXISTING PCA transform to the TEST set.
        Does NOT re-calculate mean/variance/rotation.
        """
        if self.pca is None or self.scaler is None:
            raise ValueError("You must call fit_transform (on training data) first!")

        X = dataframe.drop(columns=self.feature_cols)
        y = dataframe['Label']

        # 1. Standardize (Transform ONLY - use stats from Train)
        X_scaled = self.scaler.transform(X)

        # 2. PCA (Transform ONLY - use rotation from Train)
        principal_components = self.pca.transform(X_scaled)

        # 3. Return Result
        col_names = [f'PC{i+1}' for i in range(self.n_components)]
        result_df = pd.DataFrame(data=principal_components, columns=col_names)
        result_df['Label'] = y.values
        
        return result_df

    def plot_2d_scatter(self, pca_df: pd.DataFrame, save_path: str):
        """
        Plots the first 2 components (PC1 vs PC2).
        """
        # Calculate variance info for the plot title
        var = self.pca.explained_variance_ratio_
        pc1_var = var[0] * 100
        pc2_var = var[1] * 100
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x='PC1', y='PC2', 
            hue='Label', 
            data=pca_df, 
            palette={'Cat': 'blue', 'Dog': 'orange'},
            alpha=0.6
        )
        
        plt.title(f'PCA: 2D Projection (Train Data)\nPC1: {pc1_var:.1f}% | PC2: {pc2_var:.1f}%')
        plt.xlabel(f'Principal Component 1')
        plt.ylabel(f'Principal Component 2')
        plt.legend(title='Class')
        plt.grid(True, linestyle='--', alpha=0.5)

        if save_path:
            full_path = os.path.join(save_path, "pca_scatter.png")
            plt.savefig(full_path, dpi=300)
            print(f"Scatter plot saved to {full_path}")
        
        plt.show()

    def plot_variance_bar_chart(self, save_path: str):
        """
        Plots a Bar Chart (Scree Plot) of explained variance per component.
        """
        if self.pca is None:
            return

        # Get variance ratios
        var_ratio = self.pca.explained_variance_ratio_ * 100
        
        # Create labels (PC1, PC2, etc.)
        # Limit to top 20 for readability if n_components is large
        n_plot = min(len(var_ratio), 20)
        labels = [f'PC{i+1}' for i in range(n_plot)]
        
        plt.figure(figsize=(12, 6))
        
        # Bar Chart
        bars = plt.bar(labels, var_ratio[:n_plot], color='teal', alpha=0.7)
        
        # Add a Cumulative Line
        cumulative_var = np.cumsum(var_ratio[:n_plot])
        plt.plot(labels, cumulative_var, color='red', marker='o', linestyle='-', linewidth=2, label='Cumulative Variance')

        # Formatting
        plt.ylabel('Explained Variance (%)')
        plt.title(f'PCA Variance Explained (Top {n_plot} Components)')
        plt.ylim(0, max(cumulative_var) + 5) # Scale y-axis to fit cumulative
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # Add text labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

        if save_path:
            full_path = os.path.join(save_path, "pca_variance_bar.png")
            plt.savefig(full_path, dpi=300)
            print(f"Variance plot saved to {full_path}")
            
        plt.show()
 
    