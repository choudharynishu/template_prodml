import pickle as pk
from pathlib import Path

from code.pipeline.model import build_model
from config.config import settings

from loguru import logger

class ModelService:
    def __init__(self):
        self.model = None

    def load_model(self, model_name=settings.model_name):
        logger.info(f"Checking the existence of model config file at {settings.model_path}/{model_name}")
        model_path = Path(f'{settings.model_path}/{model_name}')

        if not model_path.exists():
            logger.warning(f"Model at {settings.model_path}/{model_name} was not found -> building {settings.model_name}")
            build_model()
        logger.info(f"Model {settings.model_path}/{model_name} exists! -> loading model configuration file!")
        self.model = pk.load(open(model_path, 'rb'))

    def predict(self, input):
        logger.info(f"Making predictions")
        return self.model.predict(input)

ml_svc = ModelService()
ml_svc.load_model(settings.model_name)
