# all required packages.
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from flask_bootstrap import Bootstrap


app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
	df = pd.read_csv('final_books_data.csv')
	# df.rename(columns={"Wow... Loved this place.": "Review", "1": "sentiment"}, inplace=True)
	# Features and Labels
	data = df[['bookTitle', 'bookRating']]

	# Extract Feature With CountVectorizer
	cv = CountVectorizer()


	matrix = cv.fit_transform(data)  # Fit the Data
# Naive Bayes Classifier
	from sklearn.neighbors import NearestNeighbors

	clf = NearestNeighbors(metric='cosine', algorithm='brute')
	clf.fit(matrix)
# Alternative Usage of Saved Model
	joblib.dump(clf, 'model.pkl')
	NB_spam_model = open('model.pkl', 'rb')
	clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		review = request.form['bookTitle']
		data = [review]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('home.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)