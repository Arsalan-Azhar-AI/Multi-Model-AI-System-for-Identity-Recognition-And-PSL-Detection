import pandas as pd
import os
from shutil import copyfile
import cv2
import numpy as np
from src.cnnClassifier import logger
import shutil
from src.cnnClassifier.entity.config_entity import DataPreprocessingConfig

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def get_celeba_images(self):
        data=pd.read_csv(self.config.data)
        # Select male and female images
        male_images = data[data['Male'] == 1]['image_id'].tolist()
        female_images = data[data['Male'] == -1]['image_id'].tolist()

        image_folder1=self.config.image_folder1

        # Copy male images
        for img in male_images[:1000]:  # Use first 1000 images
            img_path = os.path.join(image_folder1, img)  # Full path of the image
            img_path=cv2.imread(img_path)
            img_path=cv2.resize(img_path,(224,224))
            resized_image_rgb = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
            img_path=np.array(resized_image_rgb)
            save_path = os.path.join(self.config.male_path, f"um-{img}")  # Destination path
            cv2.imwrite(save_path, img_path)
        # Copy female images
        for img in female_images[:1000]:  # Use first 1000 images
            img_path = os.path.join(image_folder1, img)  # Full path of the image
            img_path=cv2.imread(img_path)
            img_path=cv2.resize(img_path,(224,224))
            resized_image_rgb = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
            img_path=np.array(resized_image_rgb)
            save_path = os.path.join(self.config.female_path, f"uf-{img}")  # Destination path
            cv2.imwrite(save_path, img_path)
        logger.info(f"CelebA Data-preprocessing Done")
    def get_knowing_person_images(self):
        folder_name=self.config.folder_name
        for sub_folder_name1 in os.listdir(folder_name):
            sub_folder_name=os.path.join(folder_name,sub_folder_name1)
            if os.path.isdir(sub_folder_name):
                images=os.listdir(sub_folder_name)
                for img_name1 in images:
                    img_name = os.path.join(sub_folder_name, img_name1)

                    # Load image
                    image = cv2.imread(img_name)

                    # Resize image (if necessary)
                    resized_image = cv2.resize(image, (224, 224))
                    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                    resized_image=np.array(resized_image_rgb)

                    #image=resized_image.convert('RGB')
                    '''
                    # Adjust brightness and contrast
                    adjusted_image = cv2.convertScaleAbs(resized_image, alpha=1.3, beta=40)  # alpha is contrast, beta is brightness

                    # Apply sharpening
                    kernel = np.array([[0, -1, 0],
                                        [-1, 5,-1],
                                        [0, -1, 0]])
                    sharpened_image = cv2.filter2D(adjusted_image, -1, kernel)

                    # Denoise the image
                    denoised_image = cv2.fastNlMeansDenoisingColored(sharpened_image, None, 10, 10, 7, 21)
                    '''
                    # Save the enhanced image

                    save_path = os.path.join(sub_folder_name, f'enhance-{img_name1}')
                    cv2.imwrite(save_path,resized_image)
                    os.remove(img_name)
        
    def knowing_person_images2(self):
        main_folder = self.config.folder_name
        for sub_dirs in os.listdir(main_folder):
            sub_dir_path = os.path.join(main_folder, sub_dirs)
            dest_dir_path = os.path.join(self.config.DATASET, sub_dirs)  # Define the destination path

            # Check if the destination folder exists
            if os.path.exists(dest_dir_path):
                shutil.rmtree(dest_dir_path)  # Remove the existing destination folder
                logger.info(f"Removed existing folder '{dest_dir_path}'")

            shutil.move(sub_dir_path, self.config.DATASET)  # Move the source folder to destination
            logger.info(f"Moved '{sub_dir_path}' to '{self.config.DATASET}'")

        logger.info(f"Knowing Person Data-preprocessing Done")




        