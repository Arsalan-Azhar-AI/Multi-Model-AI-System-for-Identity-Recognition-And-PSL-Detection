from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.data_preprocessing import DataPreprocessing
from src.cnnClassifier import logger


STAGE_NAME = "Data Preprocessing stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):

        try:
            config = ConfigurationManager()
            data_preprocessing_config = config.get_data_preprocessing_config()
            data_peprocessing = DataPreprocessing(config=data_preprocessing_config)
            data_peprocessing.get_celeba_images()
            data_peprocessing.get_knowing_person_images()
            data_peprocessing.knowing_person_images2()
        except Exception as e:
            raise e




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

