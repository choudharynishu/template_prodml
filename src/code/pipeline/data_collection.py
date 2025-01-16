import pandas as pd

from config.config import engine
from db.db_model import FMCG
from sqlalchemy import select
from loguru import logger


def load_data_from_db():
    """This function is used to load and return the data as
       a Pandas Dataframe from the provided database"""
    logger.info(f'{'Extracting the table from database'}')
    query = select(FMCG)
    return pd.read_sql(query, engine)
