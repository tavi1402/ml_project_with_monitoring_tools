import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from category_encoders import CountEncoder
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_transformer_object(self, numeric_cols, count_enc_cols, one_hot_enc_cols):
        '''
        This function is responsible for creatubg the data preprocessor object
        '''
        try:
            numeric_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())
            ])


            categorical_pipeline_1 = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('count_encoder', CountEncoder())
            ])


            categorical_pipeline_2 = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder())
            ])


            preprocessor = ColumnTransformer(
                transformers=[
                    ('numeric_pipeline', numeric_pipeline, numeric_cols),
                    ('categorical_pipeline_1', categorical_pipeline_1, count_enc_cols),
                    ('categorical_pipeline_2', categorical_pipeline_2, one_hot_enc_cols)
            ])


            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, raw_data_path):

        try:
            df_raw = pd.read_csv(raw_data_path)

            logging.info("Completed reading the raw data for further preprocessing")


            logging.info("Starting Preprocessing")

            df_raw.drop(columns = ['ID','Application_Process_Day','Application_Process_Hour', 'Mobile_Tag'], inplace=True)

            def replace_unwanted_chars(value):
                if isinstance(value, str):
                    return re.sub(r'[$#@,]', '', value)
                return value

            df_raw = df_raw.applymap(replace_unwanted_chars)
            
            potential_numerical_columns = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Population_Region_Relative',
                                           'Age_Days', 'Employed_Days', 'Registration_Days', 'ID_Days', 'Score_Source_3']
            
            potential_categorical_columns = ['Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own', 'Homephone_Tag', 'Workphone_Working']

            for col in potential_categorical_columns:
                df_raw[col] = df_raw[col].replace({1: 'Yes', 0: 'No'})

            for column in potential_numerical_columns:
                df_raw[column] = pd.to_numeric(df_raw[column], errors='coerce')

            df_raw['Client_Gender'].replace('XNA', df_raw['Client_Gender'].mode().iloc[0], inplace=True)

            df_raw['Accompany_Client'].replace('', df_raw['Accompany_Client'].mode().iloc[0], inplace=True)

            percent_missing = (df_raw.isnull().sum() / len(df_raw)) * 100

            columns_to_drop = percent_missing[percent_missing > 35].index

            df_raw.drop(columns=columns_to_drop, inplace=True)

            def handle_outliers(df):
                # Function to cap outliers based on IQR
                def cap_outliers(series, lower_bound, upper_bound):
                    series = np.where(series > upper_bound, upper_bound, series)
                    # series = np.where(series < lower_bound, lower_bound, series)
                    return series

                # Population_Region_Relative: Outliers near 0 and 100
                Q1 = df['Population_Region_Relative'].quantile(0.25)
                Q3 = df['Population_Region_Relative'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df['Population_Region_Relative'] = cap_outliers(df['Population_Region_Relative'], lower_bound, upper_bound)

                # Employed_Days: Heavily right-tailed with impossible values
                Q1 = df['Employed_Days'].quantile(0.25)
                Q3 = df['Employed_Days'].quantile(0.75)
                IQR = Q3 - Q1
                upper_bound = Q3 + 1.5 * IQR
                df = df[df['Employed_Days'] <= upper_bound]

                # Score_Source_2: Right-tailed with outliers near 100
                Q1 = df['Score_Source_2'].quantile(0.25)
                Q3 = df['Score_Source_2'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df['Score_Source_2'] = cap_outliers(df['Score_Source_2'], lower_bound, upper_bound)

                return df

            df_raw = handle_outliers(df_raw)

            columns_to_drop = ['Child_Count', 'Client_Family_Members', 'Credit_Bureau', 'Bike_Owned', 'House_Own', 'Accompany_Client']

            df_raw.drop(columns=columns_to_drop, inplace=True)

            X = df_raw.drop('Default', axis=1)
            y = df_raw['Default']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

            numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            count_enc_cols = []
            one_hot_enc_cols = []

            for col in categorical_cols:
                if len(X_train[col].value_counts()) > 2:
                    count_enc_cols.append(col)
                else:
                    one_hot_enc_cols.append(col)

            

            preprocessing_obj = self.get_data_transformer_object(numeric_cols, count_enc_cols, one_hot_enc_cols)

            
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)
            

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Saved preprocessing object")

            
            logging.info("Resampling imbalanced data")

            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_transformed, y_train)


            logging.info("Saving train and test data")

            train_data = pd.concat([pd.DataFrame(X_train_smote), pd.DataFrame(y_train_smote)], axis=1)
            train_data.to_csv(self.data_transformation_config.train_data_path, index=False)

            test_data = pd.concat([pd.DataFrame(X_test_transformed), pd.DataFrame(y_test)], axis=1)
            test_data.to_csv(self.data_transformation_config.test_data_path, index=False)


            return (
                X_train_smote,
                X_test_transformed,
                y_train_smote,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        


# if __name__ == "__main__":
    
#     data_transformation = DataTransformation()
#     X_train, X_test, y_train, y_test, preprocessor_obj_file_path = data_transformation.initiate_data_transformation('artifacts\\data.csv')