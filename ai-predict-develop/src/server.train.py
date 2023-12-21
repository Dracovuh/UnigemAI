from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import schedule
import time
import pandas as pd
import pytz
from constants import Config, QueryBuilder
from service.train import training
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from utils.logger import logger

DB = Config.Postgres
database_uri = f"postgresql+psycopg2://{DB.user}:{DB.password}@{DB.host}:{DB.port}/{DB.database}?options=-c search_path={DB.schema}"
engine = create_engine(database_uri)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = database_uri
db = SQLAlchemy(app)

utc_timezone = pytz.utc

def job():
	try:
		logger.info("Training job is running.")
		previousDate = datetime.now() - timedelta(days=2)
		fromDate = previousDate.replace(hour=0, minute=0, second=0, microsecond=0)
		toDate = previousDate.replace(hour=23, minute=59, second=59, microsecond=999999)

		queryString = db.text(QueryBuilder.TrainData)
		params = {'fromDate':fromDate, 'toDate': toDate }     

		df = pd.read_sql_query(sql = queryString, params = params, con = engine)
		# df = pd.read_csv('src\data\ckd-20231204-094300.csv')

		training(df, '15m')
		training(df, '1h')
		training(df, '6h')
		training(df, '24h')

	except Exception as e:
		logger.error(e)

schedule.every().day.at("00:00").do(job).tag("daily_job")
# job()

while True:
	schedule.run_pending()
	time.sleep(1)
