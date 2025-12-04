import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils_dataframe import create_dataframe
from sklearn.decomposition import PCA 
import os
filepath = os.path.join(os.getcwd(),'Data Files','ClimateDataBasel.csv')
main_df = create_dataframe(filepath=filepath)
import numpy as np

columns_to_keep= [
        "temp(mean)",
        "rel_humid(mean)",
        "precipitation_total",
        "sunshine_duration",
        "snowfall_amount",
        "sea_level_pressure(mean)",
        "wind_gust(mean)",
        "wind_speed(mean)"
            ]

def z_score_scaling(df:pd.DataFrame):
    s_scaler = StandardScaler()
    transformed = s_scaler.fit_transform(df)
    transformed_df = pd.DataFrame(transformed, columns=df.columns)
    return transformed_df

def keep_selected_columns(df: pd.DataFrame,columns_to_keep:list) -> pd.DataFrame:
    """
    Creates a new DataFrame retaining only the specific features selected 
    for clustering analysis.
    """
    
    
    # 2. Select only these columns
    # We use a copy() to avoid SettingWithCopy warnings later in the pipeline
    df_reduced = df[columns_to_keep].copy()
    
    print(f"Original shape: {df.shape}")
    print(f"Reduced shape:  {df_reduced.shape}")
    print(f"Features retained: {df_reduced.columns.tolist()}")
    
    return df_reduced


def perform_pca(df_scaled: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Applies PCA to reduced the dimensions of the scaled data.
    Returns the transformed dataframe (PC1, PC2, etc.).
    """
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df_scaled)
    
    # Create a nice DataFrame for the results
    columns = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(data=pca_data, columns=columns)
    
    # Critical Analysis Output (Print this for your report!)
    explained_variance = pca.explained_variance_ratio_
    total_variance = np.sum(explained_variance) * 100
    
    print(f"\n--- PCA Analysis (n={n_components}) ---")
    for i, var in enumerate(explained_variance):
        print(f"PC{i+1} explains: {var*100:.2f}% of variance")
    print(f"Total Variance Explained: {total_variance:.2f}%")
    
    return df_pca



# reduced_df = keep_selected_columns(main_df)
# scaled_df = z_score_scaling(reduced_df)

# scaled_df.to_csv(os.path.join(os.getcwd(),'Data Files','Climate_Data_Scaled_Removed.csv'))


#reduced_df = keep_selected_columns(main_df)
scaled_df = z_score_scaling(main_df)

scaled_df.to_csv(os.path.join(os.getcwd(),'Data Files','Climate_Data_Scaled.csv'))