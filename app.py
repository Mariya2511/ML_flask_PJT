import json

from flask import Flask, jsonify, request
import dill
import pandas as pd
import os
dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
from flask import Flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime



app = Flask(__name__)
model = None


handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
	# load the pre-trained model
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)
	print(model)
	return model

modelpath = "pipeline.dill"
load_model(modelpath)


@app.route('/')
def index():
    return "Start prediction http://127.0.0.1:5000/prediction"


@app.route('/prediction', methods=['GET', 'POST'])

def form_example():
    # handle the POST request
    if request.method == 'POST':
        data = request.form.get('comment')
        comment = {"Comment": data}
        comment = json.dumps(comment)
        comment = json.loads(comment)
        if comment["Comment"]:
           comment = comment['Comment']
        with open('pipeline.dill', 'rb') as in_strm:

        	model = dill.load(in_strm)
        preds = model.predict_proba(pd.DataFrame({'Data': [comment]}))

        return '''
                <h1>The comment is toxic with probability in: {}</h1>'''.format(preds)

    # otherwise handle the GET request
    return '''
           <form method="POST">
               <div><label>Comment: <input type="text" name="comment"></label></div>
               <input type="submit" value="Submit">
           </form>'''

#def predict():


	#data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	#comment = 'Did not attempt to load JSON data because the request Content-Type was no application/json python'

	#request_json = request.get_json()

	#if request_json["comment"]:
	#	comment = request_json['comment']

	#print(comment)
	#with open('pipeline.dill', 'rb') as in_strm:
	#	model = dill.load(in_strm)
	#preds = model.predict_proba(pd.DataFrame({"comment": [comment]}))
	#data["predictions"] = preds[:, 1][0]
	#data["comment"] = comment
	# indicate that the request was a success
	#data["success"] = True
	#print('OK')

	# return the data dictionary as a JSON response
	#return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)