import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting train and test data.")
            x_train, y_train, x_test, y_test = (train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1])

            models = {
                'CatBoostRegressor': CatBoostRegressor(verbose=False),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'GradientBoostRegressor': GradientBoostingRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'XGBRegressor': XGBRegressor()
            }
            params = {
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report:dict = evaluate_model(x_train, y_train, x_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found.')
            logging.info("Best model found on training and testing dataset.")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(x_test)
            model_accuracy = r2_score(y_test, predicted)
            logging.info(f"Best Model name {best_model_name} with score {model_accuracy}.")

            return model_accuracy
        except Exception as e:
            raise CustomException(e, sys)