from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath

from loguru import logger
from sqlalchemy import create_engine


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='config/.env',
                                      env_file_encoding='utf-8',)
    model_path: DirectoryPath
    model_name: str
    log_filename: str
    log_level: str
    db_conn_str: str
    fmcg_table_name: str


settings = Settings()
logger.add(settings.log_filename, level=settings.log_level,  )
engine = create_engine(settings.db_conn_str)
