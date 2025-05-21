import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf

dataset=tf.keras.preprocessing.image_dataset_from_directory(
    "artifacts/data_preprocessing/DATASET",
    shuffle=True,
    image_size=(224,224),
    batch_size=32
)
class_names=dataset.class_names


class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        #return (class_names[np.argmax(result)])
        result=class_names[np.argmax(result)]

        if result[:2] == 'M(':
                    return ({
                    "gender": "Male",
                    "status": f"He is knowing person. \nHis name is {result[2:-1]}"
                    })
        elif result[:2] == 'F(':
            return ({
                    "gender": "Female",
                    "status": f"She is knowing one. \nHer name is {result[2:-1]}"
                    })
        elif result == 'male':
                    return ({
                    "gender": "Male",
                    "status": f"He is Unknown person."
                    })
        elif result == 'female':
                return ({
                "gender": "Female",
                "status": f"She is Unknown."
                })
        else:
            return ({
            "gender": "Not Recognize.",
            "status": f"Not Recognize."
            })
