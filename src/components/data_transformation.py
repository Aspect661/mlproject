import os, sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from exception.exception import CustomException
from logger.logger import logging
from utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        """
        Responsible for returning the data transformer object
        """
        try:
            numerical_columns  = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch", 
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')), 
                    ('scaler', StandardScaler())
                ]
            )

            logging.info('Numerical columns scaling initiated')

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'most_frequent')),
                    ('onehot', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Categorical columns encoding initiated')

            logging.info(f"Categorical columns {categorical_columns}")
            logging.info(f"Numerical columns {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers = [
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ]
            )
            
            logging.info('Data transformation pipeline initiated')
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        logging.info("Initiating data transformation")
        
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Read the train and test data as dataframe')

            logging.info('Obtaining the data transformer object')
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns  = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(target_column_name, axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(target_column_name, axis = 1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info('Transforming the train and test dataframe')
            input_feature_train_array = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_array, target_feature_train_df.values]
            test_arr = np.c_[input_feature_test_array, target_feature_test_df.values]

            logging.info('Saving the preprocessor object')

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )
            

        except Exception as e:
            raise CustomException(e, sys)