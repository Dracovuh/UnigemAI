python 3.11.6

install lib
    pip install -r requirements.txt
run server
    pm2 start src\server.train.py 
    pm2 start src\server.predict.py
