import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import tensorflow as tf
from pathlib import Path
from src.cnnClassifier.entity.config_entity import EvaluationConfig
from src.cnnClassifier.utils.common import save_json
class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def get_training_val_test_dataset(self):
        self.dataset = tf.keras.preprocessing.image_dataset_from_directory(
            "artifacts/data_preprocessing/DATASET",
            shuffle=True,
            image_size=(224, 224),
            batch_size=32
        )
        self.train_ds, self.val_ds, self.test_ds = self.get_dataset_partitions_tf(self.dataset)
        self.train_ds = self.train_ds.cache().shuffle(10000).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.cache().shuffle(10000).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.cache().shuffle(10000).prefetch(buffer_size=tf.data.AUTOTUNE)
       
    def get_dataset_partitions_tf(self, ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        ds_size = sum(1 for _ in ds)  # Compute dataset size dynamically
        if shuffle:
            ds = ds.shuffle(shuffle_size, seed=12)
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size + val_size)
        
        return train_ds, val_ds, test_ds
    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self.get_training_val_test_dataset()
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            self.score = self.model.evaluate(self.test_ds)
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("Loss", self.score[0])
            mlflow.log_metric("Accuracy", self.score[1])

            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(self.model, "model", registered_model_name="VGG16")
            else:
                mlflow.sklearn.log_model(self.model, "model")

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path(self.config.metric_file_name), data=scores)