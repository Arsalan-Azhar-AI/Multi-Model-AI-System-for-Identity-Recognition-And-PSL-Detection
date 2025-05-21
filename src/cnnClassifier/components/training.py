import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from src.cnnClassifier.entity.config_entity import (PrepareCallbacksConfig,TrainingConfig)
class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config


    
    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    

    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
        filepath=str(self.config.checkpoint_model_filepath).replace(".h5", ".keras"),
        save_best_only=True
        

        )


    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]




class Training:
    def __init__(self, config: TrainingConfig):
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
        
    def get_base_model(self):
        # Load the base model
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        
        # Recreate the optimizer (e.g., Adam) after loading the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

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
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        # Train the model
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=5,  # Adjust the number of epochs as needed
            callbacks=callback_list
        )

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
