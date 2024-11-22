from flask import Flask, jsonify, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
app = Flask(__name__)

data = pd.read_csv('Salary_dataset.csv')

X = data[['YearsExperience']]
y = data [['Salary']]
model = LinearRegression().fit(X,y)

@app.route ('/data', method= ['GET'])
def get_data():
    return jsonify(data.to_dict(orient='records'))

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json 
    years_experience = content.get('YearsExperience', 0)
    prediction = model.predict([[years_experience]])
    return jsonify({'YearsExperience':years_experience, 'PredictedSalary':prediction[0]})

if __name__ =='__main__':
    app.run(debu=True)