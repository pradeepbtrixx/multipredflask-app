from flask import Flask,url_for,render_template,request
import pickle 
import numpy as np

app = Flask(__name__)
salmodel = pickle.load(open('salarypr_model.pickle','rb'))
lungmodel = pickle.load(open('lungpredictionmodel.pickle','rb'))
bna = pickle.load(open('BNAclassifier.pickle','rb'))

@app.route('/')
def greet():
	return render_template('main.html')

@app.route('/salpre')
def salpre():
	return render_template('salpre.html')

@app.route('/lungcpre')
def lungcpre():
	return render_template('lungcanpre.html')

@app.route('/BNA')
def BNA():
	return render_template('bna.html')
# this is salprediction function (main.html)
@app.route('/salpredict',methods=['POST'])
def salpredict():
	int_features = [int(x) for x in request.form.values()]
	values = [np.array(int_features)]
	prediction = salmodel.predict(values)
	out = prediction[0].round()
	return render_template('salpre.html',prediction_text= f"your predicted salary is - {out}")

@app.route('/lungpredict',methods=['POST'])
def lungpredict():
	int_features = [int(x) for x in request.form.values()]
	values = [np.array(int_features)]
	prediction = lungmodel.predict(values)
	out = prediction[0]
	result = ''
	if out == 0:
		result = 'your perfectly alright'
	else:
		result =  'sorry your cancer patient'

	return render_template('lungcanpre.html',prediction_text= result)

@app.route('/bnapre',methods=['POST'])
def bnapre():
	int_features = [float(x) for x in request.form.values()]
	values = [np.array(int_features)]
	prediction = bna.predict(values)
	out = prediction[0]
	return render_template('bna.html',prediction_text= f"your output is : {out}")



if __name__ == "__main__":
	app.run(debug=True)

