import PIL.Image as Image
import cv2
import numpy as np
import matplotlib.pylab as plt
import os
import seaborn as sns
import pandas as pd
import random
from sklearn.model_selection import train_test_split


class initialization:
    def __init__(self, path: str) -> None:
        self.path = path
        self.image_df = pd.DataFrame()

    def create_dataframe(self) -> pd.DataFrame:
        images = []
        labels = []
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

        try:
            for foldr in os.listdir(self.path):
                # Skip hidden files like .DS_Store
                if foldr.startswith('.'): continue
                
                folder_full = os.path.join(self.path, foldr)
                if os.path.isdir(folder_full):
                    for filee in os.listdir(folder_full):
                        file_ext = os.path.splitext(filee.lower())[1]
                        if file_ext in valid_ext:
                            # We store relative path "Cat/0.jpg"
                            images.append(os.path.join(filee))
                            labels.append(foldr)
        except Exception as e:
            print(f'Error: {e}')

        self.image_df = pd.DataFrame({
            'Images': images,
            'Labels': labels
        })

        # self.image_df,_ = train_test_split(
        #     self.image_df, 
        #     train_size=5000, 
        #     stratify=self.image_df['Labels'], 
        #     random_state=42
        # )
        return pd.DataFrame(self.image_df)

    def split_data(self, train_size=0.7, val_size=0.15, test_size=0.15, random_seed=42):
        """
        Splits the dataframe into Train, Validation, and Test sets.
        Ensures the classes (Cats/Dogs) are balanced in each split using stratify.
        """

        if not np.isclose(train_size + val_size + test_size, 1.0):
            print(f"Warning: Ratios sum to {train_size + val_size + test_size}, not 1.0")


        train_val_df, test_df = train_test_split(
            self.image_df, 
            test_size=test_size, 
            stratify=self.image_df['Labels'], 
            random_state=random_seed
        )

        relative_val_size = val_size / (train_size + val_size)
        
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=relative_val_size, 
            stratify=train_val_df['Labels'], 
            random_state=random_seed
        )

        print("-" * 30)
        print(f"Data Splitting Complete:")
        print(f"Train Set:      {len(train_df)} images ({train_size*100}%)")
        print(f"Validation Set: {len(val_df)} images ({val_size*100}%)")
        print(f"Test Set:       {len(test_df)} images ({test_size*100}%)")
        print("-" * 30)

        return train_df, val_df, test_df

    def view_extreme_images(self, save_path: str):
        """
        Fixed version: correctly joins paths and handles errors
        """
        min_area = float('inf')
        max_area = 0
        min_img_path = None
        max_img_path = None
        min_dims = (0, 0)
        max_dims = (0, 0)
        
        print("Scanning dataset for image dimensions (this may take a moment)...")
        
        # Iterate over the dataframe rows
        for idx, row in self.image_df.iterrows():
            # Correctly join the base path with the relative image path
            full_path = os.path.join(self.path,row['Labels'], row['Images'])
            
            try:
                # Open lazily with PIL
                with Image.open(full_path) as img:
                    width, height = img.size
                    area = width * height
                    
                    if area < min_area:
                        min_area = area
                        min_img_path = full_path
                        min_dims = (width, height)
                        
                    if area > max_area:
                        max_area = area
                        max_img_path = full_path
                        max_dims = (width, height)
            except Exception as e:
                # If image is corrupt, just skip
                pass

        print(f"Stats calculated!")
        print(f"Smallest Image: {min_dims} (WxH)")
        print(f"Largest Image:  {max_dims} (WxH)")

        # Create the plot
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        if min_img_path:
            img_min = cv2.imread(min_img_path)
            print(f"minimum image = {min_img_path}")
            if img_min is not None:
                img_min = cv2.cvtColor(img_min, cv2.COLOR_BGR2RGB)
                axes[0].imshow(img_min)
                axes[0].set_title(f"Smallest Image:{min_img_path}\n{min_dims[0]}x{min_dims[1]}")
                axes[0].axis('off')
        
        if max_img_path:
            img_max = cv2.imread(max_img_path)
            print(f"maximum image = {max_img_path}")
            if img_max is not None:
                img_max = cv2.cvtColor(img_max, cv2.COLOR_BGR2RGB)
                axes[1].imshow(img_max)
                axes[1].set_title(f"Largest Image \n{max_dims[0]}x{max_dims[1]}")
                axes[1].axis('off')

        plt.tight_layout()
        save_file = os.path.join(save_path, "extreme_dimensions.png")
        plt.savefig(save_file, dpi=300)
        print(f"Saved plot to {save_file}")
        plt.show()

if __name__ == "__main__":
    # Adjust path as necessary
    base_path = os.path.join(os.getcwd(),"Task 2- Image Classification")
    folderpath = os.path.join(base_path,"Data","PetImages")
    
    # 1. Initialize and Load
    image_prep = initialization(path=folderpath)
    labels_df = image_prep.create_dataframe()
    
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

    # 2. Exploratory Analysis
    plots_path = os.path.join(base_path,"plots")
    image_prep.view_extreme_images(save_path=plots_path)