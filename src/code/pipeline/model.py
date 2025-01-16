import pickle as pk
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from code.pipeline.preparation import prepare_data
from config.config import settings
from loguru import logger


def build_model():
    """
    This function is used to load, call the prepare data,
    training, and testing functions
    """

    logger.info(f'{'Starting up model building pipeline'}')
    data = prepare_data()
    # Get x - feature variables and y - target variable
    x, y = _get_x_y(data)
    # Divide the dataset into train and test sets
    train_X, test_X, train_y, test_y = _split_train_test(x, y)
    # Scale the features
    train_X, test_X = _scale_features(train_X, test_X)
    # Define the model
    rf = _train_model(train_X, train_y)
    # Get test performance score
    score = _evaluate_model(rf, test_X, test_y)
    print(score)
    _save_model(rf)


def _get_x_y(data: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    This function divides the processed into features and target variables.

    :param data: pd.DataFrame, processed dataframe that needs to be divided
    :return: (x,y) (pd.DataFrame, pd.Series), feature and target variables
    """

    x = data.drop(columns=['Date',
                           'Sales_Volume',])
    y = data['Sales_Volume']
    logger.info(f'Defining X and y variables. \nX vars: {x.columns}'
                f'\ny var: {y.name}')
    return x, y


def _split_train_test(
        x: pd.DataFrame,
        y: pd.Series,
):
    """
    This function divides the give feature variables and corresponding target
    variable into training and test sets. The obtained test set is of size
    test_size (default=0.20) times the size of the original dataset.

    :param x: pd.DataFrame, Feature variables
    :param y: pd.Series, Target variable
    :return: Feature and Target variables for train and test sets
    """

    logger.info(f'{'Splitting the data into train and test sets'}')
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2)
    return train_X, test_X, train_y, test_y


def _scale_features(
        train_X: pd.DataFrame,
        test_X: pd.DataFrame,
) -> (pd.DataFrame, pd.DataFrame):
    """
    This function is used to scale the feature variables of the train
    and test sets. The scaling transform each feature from zero to one.

    :param train_X: pd.DataFrame, feature variables from the training set
    :param test_X: pd.DataFrame, feature variables from the test set
    :return train_X, test_X: scaled features for both training and test sets
    """

    # Instantiate a minmax scaling object
    minmax_scaling = MinMaxScaler()
    # Fit training data to the instantiated object. Fitting in this context
    # means extracting minimum and maximum value for each feature using only
    # the training dataset
    minmax_scaling.fit(train_X)
    # Apply the learned scaling to the training set
    train_X = minmax_scaling.transform(train_X)
    # Apply the learned scaling to the test set
    test_X = minmax_scaling.transform(test_X)
    return train_X, test_X


def _train_model(
        train_X: pd.DataFrame,
        train_y: pd.Series
) -> RandomForestRegressor:
    """
    This function is used to the train a Random Forest Regression model.

    This function is used to train a Random Forest Regression model on the
    given dataset. The training first involves hyperparameter tuning, for which
    we utilized the grid search method. The tuned hyperparameters are, number
    of estimators and maximum depth for any given estimator. Further, a 5-fold
    stratified cross validation is also applied while tuning. The tuned models
    are evaluated using R-squared values.

    :param train_X: pd.DataFrame, feature variables
    :param train_y: pd.Series, target variable
    :return model: RandomForestRegressor, trained Random Forest model
    """

    logger.info(f'{'Training the model with Hyperparameter Tuning'}')
    # Hyperparameters and the corresponding search space
    parameters = {'n_estimators': [100, 200, 300],
                  'max_depth': [3, 6, 9, 12]}
    # Instantiate the cross validation object
    kfold = KFold()
    # Instantiate the Grid Search object
    grid = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid=parameters,
        scoring='r2',
        cv=kfold,
    )
    # Hyperparameter tuning using instantiated grid search method
    grid.fit(train_X, train_y)
    model = grid.best_estimator_
    print(model)
    logger.info(f'{'Model best score {grid.best_score_}'}')
    return model


def _evaluate_model(
        model: RandomForestRegressor,
        test_x: pd.DataFrame,
        test_y: pd.Series,
) -> float:
    """
    This function is used to evaluate the trained Random forest model.
    The trained model is evaluated using scaled features and target variable
    from the test set.

    :param model: RandomForestRegressor, trained model
    :param test_x: pd.DataFrame, feature variables
    :param test_y: pd.Series, target variable
    :return model_score: float, R-squared value of the trained model
    """

    model_score = model.score(test_x, test_y)
    logger.info(f'{'Evaluating the model. Score={model_score}'}')
    return model_score


def _save_model(model: RandomForestRegressor):
    """
    Save the trained model into a specified directory.

    :param model: RandomForestRegressor, the model to save
    :return: None
    """
    model_path = f'{settings.model_path}/{settings.model_name}'
    logger.info(f'Saving a model to directory: {model_path}')
    with open(model_path, 'wb') as model_files:
        pk.dump(model, model_files)
