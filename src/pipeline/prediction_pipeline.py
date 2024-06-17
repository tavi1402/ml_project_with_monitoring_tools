import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            return predictions
        
        except Exception as e:
            raise CustomException(e, sys)

# Class to map inputs given to HTML with backend
class CustomData:
    def __init__(self,
        Client_Income: float,
        Car_Owned: str,
        Active_Loan: str,
        Credit_Amount: float,
        Loan_Annuity: float,
        Client_Income_Type: str,
        Client_Education: str,
        Client_Marital_Status: str,
        Client_Gender: str,
        Loan_Contract_Type: str,
        Client_Housing_Type: str,
        Population_Region_Relative: float,
        Age_Days: float,
        Employed_Days: float,
        Registration_Days: float,
        ID_Days: float,
        Homephone_Tag: str,
        Workphone_Working: str,
        Client_Occupation: str,
        Cleint_City_Rating: float,
        Client_Permanent_Match_Tag: str,
        Client_Contact_Work_Tag: str,
        Type_Organization: str,
        Score_Source_2: float,
        Score_Source_3: float,
        Phone_Change: float
    ):
        self.Client_Income = Client_Income
        self.Car_Owned = Car_Owned
        self.Active_Loan = Active_Loan
        self.Credit_Amount = Credit_Amount
        self.Loan_Annuity = Loan_Annuity
        self.Client_Income_Type = Client_Income_Type
        self.Client_Education = Client_Education
        self.Client_Marital_Status = Client_Marital_Status
        self.Client_Gender = Client_Gender
        self.Loan_Contract_Type = Loan_Contract_Type
        self.Client_Housing_Type = Client_Housing_Type
        self.Population_Region_Relative = Population_Region_Relative
        self.Age_Days = Age_Days
        self.Employed_Days = Employed_Days
        self.Registration_Days = Registration_Days
        self.ID_Days = ID_Days
        self.Homephone_Tag = Homephone_Tag
        self.Workphone_Working = Workphone_Working
        self.Client_Occupation = Client_Occupation
        self.Cleint_City_Rating = Cleint_City_Rating
        self.Client_Permanent_Match_Tag = Client_Permanent_Match_Tag
        self.Client_Contact_Work_Tag = Client_Contact_Work_Tag
        self.Type_Organization = Type_Organization
        self.Score_Source_2 = Score_Source_2
        self.Score_Source_3 = Score_Source_3
        self.Phone_Change = Phone_Change

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Client_Income": [self.Client_Income],
                "Car_Owned": [self.Car_Owned],
                "Active_Loan": [self.Active_Loan],
                "Credit_Amount": [self.Credit_Amount],
                "Loan_Annuity": [self.Loan_Annuity],
                "Client_Income_Type": [self.Client_Income_Type],
                "Client_Education": [self.Client_Education],
                "Client_Marital_Status": [self.Client_Marital_Status],
                "Client_Gender": [self.Client_Gender],
                "Loan_Contract_Type": [self.Loan_Contract_Type],
                "Client_Housing_Type": [self.Client_Housing_Type],
                "Population_Region_Relative": [self.Population_Region_Relative],
                "Age_Days": [self.Age_Days],
                "Employed_Days": [self.Employed_Days],
                "Registration_Days": [self.Registration_Days],
                "ID_Days": [self.ID_Days],
                "Homephone_Tag": [self.Homephone_Tag],
                "Workphone_Working": [self.Workphone_Working],
                "Client_Occupation": [self.Client_Occupation],
                "Cleint_City_Rating": [self.Cleint_City_Rating],
                "Client_Permanent_Match_Tag": [self.Client_Permanent_Match_Tag],
                "Client_Contact_Work_Tag": [self.Client_Contact_Work_Tag],
                "Type_Organization": [self.Type_Organization],
                "Score_Source_2": [self.Score_Source_2],
                "Score_Source_3": [self.Score_Source_3],
                "Phone_Change": [self.Phone_Change]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)