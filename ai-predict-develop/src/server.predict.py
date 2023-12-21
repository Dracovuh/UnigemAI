
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from constants import Config, Intervals, QueryBuilder
from utils.logger import logger
# from controller.controller import predictController
from service.predict import predict
import pandas as pd

DB = Config.Postgres
database_uri = f"postgresql+psycopg2://{DB.user}:{DB.password}@{DB.host}:{DB.port}/{DB.database}?options=-c search_path={DB.schema}"
engine = create_engine(database_uri)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = database_uri
db = SQLAlchemy(app)

@app.route('/')
def hello_world():
    return jsonify(status=True)

@app.route('/api/predict', methods=['POST'])
def apiEndpoint():
    try:
        data = request.get_json()

        if 'contract' not in data and 'interval' not in data:
            error_message = {
                'status': False, 
                'message': 'Missing parameters: contract and interval'}
            return jsonify(error_message), 400 
        
        contract = data['contract']
        interval = data['interval']

        if interval not in Intervals:
            error_message = {
                'status': False, 
                'message': f'Interval not supported ({Intervals})'}
            return jsonify(error_message), 400 

        queryString = db.text(QueryBuilder.PredictData)
        params ={'contractAddress': contract, 'WS': Config.Training.WS}
        df = pd.read_sql_query(sql = queryString, params = params, con = engine)

        if df.empty:
            response = {
            'status': False,
            'message':'This contract address has not data therefore cannot predict yet.'
            }
            return jsonify(response)

        result = predict(interval=interval, data=df)

        if result['status'] == True:
            data = result['data']
            response_data = {
                "prices": data['prices'].tolist(),
                "priceChange": data['priceChange'].tolist(),
                }
            return jsonify(
                status = True,
                data = response_data
                )
        else:
            return jsonify(
                status = False,
                message = result['message']
                )
    except Exception as e:
        logger.error(e)
        return jsonify(
            status=False,
            message="Something went wrong"), 500
 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=Config.Port)
