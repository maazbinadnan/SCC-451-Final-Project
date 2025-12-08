import PIL.Image as Image
import numpy as np
import matplotlib.pylab as plt
import os
import seaborn as sns
import pandas as pd
from torchvision.transforms import v2
from torchvision.models import resnet50,ResNet50_Weights
import torch
from torchvision.io import decode_image,ImageReadMode
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

base_path =  path = os.path.dirname(__file__)

class preprocessing:
    def __init__(self, path: str) -> None:
        self.path = path
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on: {self.device}")
        
        self.weights = ResNet50_Weights.DEFAULT
        self.resnet_transforms = self.weights.transforms()
        
        self.model = resnet50(weights=self.weights)
        
        self.model.fc = torch.nn.Identity() # type: ignore
        self.model.eval()
        
        self.model.to(self.device)

    def transform_for_resnet(self,image_df:pd.DataFrame):
        feature_vector = []
        labels_vector = []
        
        print(f"Starting feature extraction on CPU for {len(image_df)} images...")
        print("Note: This may take 1-2 hours depending on your CPU speed.")
        
        with torch.no_grad():
            for idx, row in enumerate(image_df.itertuples(index=False)):
                try:
                    full_path = os.path.join(base_path, "Data", "PetImages", f"{row.Images}")
                    
                    img = decode_image(full_path, mode=ImageReadMode.RGB)
                    
                    transformed_image = self.resnet_transforms(img)
                    
                    transformed_image = transformed_image.unsqueeze(0).to(self.device)

                    features = self.model(transformed_image)

                    features_np = features.squeeze().numpy()

                    feature_vector.append(features_np)
                    labels_vector.append(row.Labels)
                    
                    if idx % 100 == 0:
                        print(f"Processed {idx}/{len(image_df)}...")

                except Exception as e:
                    print(f"Error at Row {idx} ({row.Images}): {e}")

        print("Creating final DataFrame...")
        df_result = pd.DataFrame(feature_vector)
        df_result['Label'] = labels_vector
        
        return df_result
    


if __name__ == "__main__":
    # # Adjust path as necessary
    base_path = os.path.join(os.getcwd(),"Task 2- Image Classification")
    # # folderpath = os.path.join(base_path,"Cats and Dogs - Data","PetImages")
    prep = preprocessing(path = os.path.join(base_path,"train_split.csv"))
    # feature_vector = prep.transform_for_resnet()
    # feature_vector.to_csv("train_vectors.csv")

    feature_vector_df = pd.read_csv(os.path.join(base_path,"train_vectors_all.csv"))

    #reduced_df = prep.dimensionality_reduction(dataframe=feature_vector_df,save_plot_path=os.path.join(base_path,"plots"))
    

