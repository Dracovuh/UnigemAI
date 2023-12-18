import datetime
from flask import jsonify

def predictController():
    try:
        x = datetime.datetime.now()
        # data = request.get_json()
        # result = predict(data)

        result = {
            "prices":[[x, 0.2], [x, 0.2],[x, 0.2], [x, 0.2],[x, 0.2], [x, 0.2],[x, 0.2], [x, 0.2]],
            "priceChange": -89
            }

        return jsonify(
            status = True,
            data = result
            )

    except Exception as e:
        print(e)
        return jsonify(
            message=str(e),
            error="Error processing JSON data"), 500