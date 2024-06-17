from flask import Flask, request, app, render_template
import os

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

## Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

## Route for prediction
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data = CustomData(
            Client_Income = request.form.get('Client_Income'),
            Car_Owned = request.form.get('Car_Owned'),
            Active_Loan = request.form.get('Active_Loan'),
            Credit_Amount = request.form.get('Credit_Amount'),
            Loan_Annuity = request.form.get('Loan_Annuity'),
            Client_Income_Type = request.form.get('Client_Income_Type'),
            Client_Education = request.form.get('Client_Education'),
            Client_Marital_Status = request.form.get('Client_Marital_Status'),
            Client_Gender = request.form.get('Client_Gender'),
            Loan_Contract_Type = request.form.get('Loan_Contract_Type'),
            Client_Housing_Type = request.form.get('Client_Housing_Type'),
            Population_Region_Relative = request.form.get('Population_Region_Relative'),
            Age_Days = request.form.get('Age_Days'),
            Employed_Days = request.form.get('Employed_Days'),
            Registration_Days = request.form.get('Registration_Days'),
            ID_Days = request.form.get('ID_Days'),
            Homephone_Tag = request.form.get('Homephone_Tag'),
            Workphone_Working = request.form.get('Workphone_Working'),
            Client_Occupation = request.form.get('Client_Occupation'),
            Cleint_City_Rating = request.form.get('Cleint_City_Rating'),
            Client_Permanent_Match_Tag = request.form.get('Client_Permanent_Match_Tag'),
            Client_Contact_Work_Tag = request.form.get('Client_Contact_Work_Tag'),
            Type_Organization = request.form.get('Type_Organization'),
            Score_Source_2 = request.form.get('Score_Source_2'),
            Score_Source_3 = request.form.get('Score_Source_3'),
            Phone_Change = request.form.get('Phone_Change')
        )

        pred_df = data.get_data_as_dataframe()

        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        results = ['No Default' if results[0] == 0 else 'Default']
        
        return render_template('index.html',results=results[0])


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # Changed to 8000
    app.run(host='0.0.0.0', port=port)
