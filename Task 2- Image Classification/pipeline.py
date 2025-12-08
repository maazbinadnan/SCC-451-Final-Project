import os
import pandas as pd
from dataset_initialization import initialization
from dataset_preprocessing import preprocessing
from dimensionality_reduction import PCAHandler
from Image_Clustering import ClusteringAnalysis
# Adjust path as necessary
base_path = os.path.join(os.getcwd(),"Task 2- Image Classification")


if __name__ == "__main__":
    # Adjust path as necessary
    base_path = os.path.join(os.getcwd(),"Task 2- Image Classification")
    datafolderpath = os.path.join(base_path,"Data","PetImages")
    plots_path = os.path.join(base_path,"plots")
    
    # # 1. Initialize and Load
    # image_prep = initialization(path=datafolderpath)
    # labels_df = image_prep.create_dataframe()
    
    # if not labels_df.empty:
    #     # 2. Exploratory Analysis
    #     plots_path = os.path.join(base_path,"plots")
    #     image_prep.view_extreme_images(save_path=plots_path)
        
    #     # 3. Create Splits
    #     train_df, val_df, test_df = image_prep.split_data(train_size=0.7, val_size=0.15, test_size=0.15)
        
    #     # 4. Save splits to CSV for the next stage (Feature Extraction)
    #     # It's good practice to save these so your training is reproducible!
    #     train_df.to_csv(f"{base_path}\\train_split.csv", index=False)
    #     val_df.to_csv(f"{base_path}\\val_split.csv", index=False)
    #     test_df.to_csv(f"{base_path}\\test_split.csv", index=False)
    #     print("Splits saved to CSV.")
    # else:
    #     print("No images found. Check your folderpath.")


    prep = preprocessing(path = os.path.join(base_path,"train_split.csv"))
    # feature_vector = prep.transform_for_resnet()
    # feature_vector.to_csv("train_vectors.csv")

    feature_vector_df = pd.read_csv(os.path.join(base_path,"train_vectors_all.csv"),index_col=0)

    # 1. Initialize (Use 50 components for clustering tasks, or 2 just for viz)
    pca_tool = PCAHandler(n_components=50)

    # 2. FIT on Train
    print("Fitting PCA on Training Data...")
    train_pca_df = pca_tool.fit_transform(feature_vector_df)

    
    pca_tool.plot_2d_scatter(train_pca_df, save_path=plots_path)

    # # 5. Plot Variance Bar Chart
    pca_tool.plot_variance_bar_chart(save_path=plots_path)

    #reduced_df = prep.dimensionality_reduction(dataframe=feature_vector_df,save_plot_path=plots_path)

    #do clustering
    clusterer = ClusteringAnalysis(feature_df=train_pca_df)

    clusterer.plot_elbow_method(max_k=10,save_path=plots_path)
    clusterer.plot_clustering_metrics(max_k=10,save_path=plots_path)
