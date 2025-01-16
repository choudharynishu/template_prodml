import pandas as pd
from code.pipeline.data_collection import load_data_from_db

from loguru import logger


def prepare_data() -> pd.DataFrame:
    """
    This function is used to prepare the loaded dataset after
    encoding the categorical columns.
    :return: encoded_data (pd.DataFrame) Transformed dataset ready for training
    """

    # Load the dataset
    logger.info(f'{'Started up data pre-processing pipeline'}')
    data = load_data_from_db()
    logger.info(f'{'Data Loaded from database'}')
    # Encode Store location and Product Type columns
    encoded_data = _encode_cat_columns(data)
    return encoded_data


def _encode_cat_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to encode string categorical columns as
    numerical categories

    :param data: pd.DataFrame, unprocessed dataframe with string categories
    :return: pd.DataFrame, processed dataframe with numerical categories
    """

    cat_columns = ['Product_Category', 'Store_Location']
    logger.info(f"Encoding categorical columns {cat_columns}")
    data_encoded = pd.get_dummies(data, columns=cat_columns, drop_first=True)
    return data_encoded
