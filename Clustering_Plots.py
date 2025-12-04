import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from Cluster_Preprocessing import z_score_scaling
from utils_dataframe import create_dataframe
import math

def create_plots(df: pd.DataFrame, folderpath: str, filename: str = "all_histograms.pdf"):    
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    filepath = os.path.join(folderpath, filename)
    
    columns = df.columns
    num_plots_per_page = 6
    num_pages = math.ceil(len(columns) / num_plots_per_page)

    print(f"Generating PDF with {num_pages} pages...")

    # Open the PDF context
    with PdfPages(filepath) as pdf:
        for page in range(num_pages):
            # Create figure
            fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27)) # A4 Landscape size (approx)
            axes = axes.flatten()
            
            # Determine batch
            start_idx = page * num_plots_per_page
            end_idx = min((page + 1) * num_plots_per_page, len(columns))
            current_batch = columns[start_idx:end_idx]
            
            for i, col in enumerate(current_batch):
                ax = axes[i]
                df[col].hist(ax=ax, bins=30, color='skyblue', edgecolor='black', grid=False)
                ax.set_title(f'{col}', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=8)
                
                # Scientific notation for y-axis if numbers are huge (optional but good for visibility)
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

            # Remove empty subplots
            for j in range(len(current_batch), len(axes)):
                fig.delaxes(axes[j])
                
            plt.suptitle(f"Basel Climate Data Distribution - Page {page+1}", fontsize=14)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95)) # Adjust for suptitle
            
            # SAVE the current figure to the PDF page
            pdf.savefig(fig) 
            plt.close(fig) 
            
    print(f"Successfully saved single PDF to: {filepath}")

def create_correlation_heatmap(df: pd.DataFrame, folderpath: str, filename: str = "correlation_heatmap.pdf"):
    """
    Generates a correlation heatmap of the dataframe and saves it as a single-page PDF.
    """
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    filepath = os.path.join(folderpath, filename)
    print("Generating Correlation Heatmap...")

    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Create figure - typically needs to be large for 18x18 grid
    plt.figure(figsize=(12, 10)) 
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # Generate Heatmap
    # annot=True adds the numbers to the squares
    # fmt=".2f" formats them to 2 decimal places
    # cmap='coolwarm' is standard for correlation (Red=Pos, Blue=Neg)
    sns.heatmap(corr_matrix, annot=True,mask=mask, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})

    plt.title("Feature Correlation Matrix (Basel Climate Data)", fontsize=16, pad=20)
    
    # Adjust layout to ensure labels aren't cut off
    plt.tight_layout()

    # Save to PDF
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"Successfully saved Correlation Heatmap to: {filepath}")

def create_boxplots_pdf(df: pd.DataFrame, folderpath: str, filename: str = "boxplots_outliers.pdf"):
    """
    Generates box plots for each column in the dataframe to identify outliers
    and saves them into a multi-page PDF.
    """
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    filepath = os.path.join(folderpath, filename)
    
    columns = df.columns
    num_plots_per_page = 6
    num_pages = math.ceil(len(columns) / num_plots_per_page)

    print(f"Generating Box Plots PDF with {num_pages} pages...")

    with PdfPages(filepath) as pdf:
        for page in range(num_pages):
            # Create figure
            fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27)) 
            axes = axes.flatten()
            
            # Determine batch
            start_idx = page * num_plots_per_page
            end_idx = min((page + 1) * num_plots_per_page, len(columns))
            current_batch = columns[start_idx:end_idx]
            
            for i, col in enumerate(current_batch):
                ax = axes[i]
                
                # Plot Box Plot
                # 'vert=True' makes them vertical, 'patch_artist=True' allows filling color
                df[col].plot.box(ax=ax, vert=True, patch_artist=True, 
                                 boxprops=dict(facecolor="lightgreen"))
                
                ax.set_title(f'{col}', fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Remove x-axis label (it's just "1" by default in pandas boxplot)
                ax.set_xticklabels([])

            # Remove empty subplots
            for j in range(len(current_batch), len(axes)):
                fig.delaxes(axes[j])
                
            plt.suptitle(f"Outlier Detection (Box Plots) - Page {page+1}", fontsize=14)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            
            # Save page
            pdf.savefig(fig) 
            plt.close(fig) 
            
    print(f"Successfully saved Box Plots to: {filepath}")


def create_pca_variance_plot(df_scaled: pd.DataFrame, folderpath: str, filename: str = "pca_variance_plot.pdf"):
    """
    Generates a PCA Scree Plot (Explained Variance) and saves it as a single-page PDF.
    """
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    filepath = os.path.join(folderpath, filename)
    print("Generating PCA Variance Plot...")

    # Fit PCA
    # n_components=None means keep all components so we can see the full curve
    pca = PCA(n_components=None)
    pca.fit(df_scaled)
    
    # Calculate metrics
    exp_var = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(exp_var)
    n_components = len(exp_var)

    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot Individual Variance (Bars)
    plt.bar(range(1, n_components + 1), exp_var, alpha=0.6, align='center',
            label='Individual explained variance', color='steelblue', edgecolor='black')
    
    # Plot Cumulative Variance (Line)
    plt.step(range(1, n_components + 1), cum_var, where='mid',
             label='Cumulative explained variance', color='red', linewidth=2)
    
    # Add a visual threshold line (e.g., at 95% variance)
    plt.axhline(y=95, color='green', linestyle='--', label='95% Explained Variance')
    
    plt.ylabel('Explained Variance Ratio (%)', fontsize=12)
    plt.xlabel('Principal Component Index', fontsize=12)
    plt.title('PCA Scree Plot: Explained Variance', fontsize=16, pad=20)
    plt.xticks(range(1, n_components + 1))
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()

    # Save to PDF
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"Successfully saved PCA Variance Plot to: {filepath}")

def save_metrics_to_pdf(metrics_df: pd.DataFrame, pdf_path: str, title: str):
    """
    Universal plotting function for clustering metrics.
    It automatically detects which metrics are in the DataFrame and plots them.
    """
    # Define columns to ignore in plotting (like the x-axis variable)
    x_axis_col = 'K' if 'K' in metrics_df.columns else 'Eps'
    metric_cols = [c for c in metrics_df.columns if c != x_axis_col]
    
    num_plots = len(metric_cols)
    rows = math.ceil(num_plots / 2)
    
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows))
        fig.suptitle(title, fontsize=16)
        axes = axes.flatten()
        
        for i, col in enumerate(metric_cols):
            ax = axes[i]
            # Plot logic
            ax.plot(metrics_df[x_axis_col], metrics_df[col], 'o-', label=col)
            
            # Styling
            ax.set_title(col)
            ax.set_xlabel(x_axis_col)
            ax.set_ylabel("Score")
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Visual cues for "best" direction
            if "Inertia" in col or "Davies" in col or "BIC" in col or "AIC" in col:
                ax.set_title(f"{col} (Lower is Better)")
            else:
                ax.set_title(f"{col} (Higher is Better)")

        # Hide empty subplots
        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        pdf.savefig(fig)
        plt.close()
    
    print(f"Saved plotting PDF: {pdf_path}")




filepath = os.path.join(os.getcwd(),'Data Files','ClimateDataBasel.csv')

main_df = create_dataframe(filepath=filepath)
folderpath=f"{os.getcwd()}\\plots"

scaled_df = z_score_scaling(main_df)

create_pca_variance_plot(scaled_df,folderpath)

# # #create_plots(main_df,folderpath=folderpath)
# create_correlation_heatmap(main_df,folderpath=folderpath)
# # create_boxplots_pdf(main_df,folderpath=folderpath)

