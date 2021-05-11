import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, json
import json as js

model = pickle.load(open('churn_model.pkl',  'rb'))


app = Flask(__name__)


@app.route('/prediction')
def predict_churn():
    is_male = int(request.args.get('is_male'))
    num_inters = int(request.args.get('num_inters'))
    late_on_payment = int(request.args.get('late_on_payment'))
    age = int(request.args.get('age'))
    years_in_contract = float(request.args.get('years_in_contract'))
    df = pd.DataFrame({'is_male': [is_male], 'num_inters': [num_inters], 'late_on_payment': [late_on_payment],
                       'age': [age], 'years_in_contract': [years_in_contract]})
    pred = model.predict(df)

    return str(pred[0])


@app.route('/predict_churn_bulk', methods=['POST'])
def predict_churns():
    predictions = {}
    inputs = js.loads(request.get_json())
    for sample, params in enumerate(inputs):
        df = pd.DataFrame(params, index=[0])
        pred = model.predict(df)
        predictions['sample ' + str(sample)] = int(pred[0])

    response = json.jsonify(predictions)

    return response


if __name__ == '__main__':
    port = os.environ.get('PORT')

    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()
