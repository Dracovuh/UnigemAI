
import datetime
from flask import Flask, jsonify

from service.predict import predicting

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return jsonify(message="Hello, World a!")

@app.route('/api/predict', methods=['POST'])
def predictApi():
    # predictController()
    try:
        x = datetime.datetime.now()
        # data = request.get_json()
        result = predicting()
        # result = {
        #     "prices":[[x, 0.2], [x, 0.2],[x, 0.2], [x, 0.2],[x, 0.2], [x, 0.2],[x, 0.2], [x, 0.2]],
        #     "priceChange": -89
        #     }
        return jsonify(
            status = True,
            data = result
            )

    except Exception as e:
        print(e)
        return jsonify(
            message=str(e),
            error="Error processing JSON data"), 500

if __name__ == '__main__':
    app.run(debug=False)
