from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
	df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
	# Features and Labels
	x = df.iloc[:, :3].values
	y = df.iloc[:, -1:].values

	#Logistic Regression Classifier
	from sklearn.ensemble import AdaBoostRegressor
	adr = AdaBoostRegressor().fit(x, y)
	#Alternative Usage of Saved Model
	joblib.dump(adr, 'model.pkl')
	model = open('model.pkl','rb')
	clf = joblib.load(model)

	if request.method == 'POST':
		int_features = [float(x) for x in request.form.values()]
		final_features = [np.array(int_features)]
		prediction = clf.predict(final_features)
		my_prediction= prediction.astype(int)
	return render_template('home.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)