from flask import Flask
from datetime import datetime
import pandas as pd
import pymysql

app = Flask(__name__)

# Configuration to MySQL
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    database='pertanian'
)
cursor = conn.cursor()

# Train data and parse into dataframe
def getDataframe(comodity, dateRange):
    sql = "SELECT tanggal, harga_current FROM pertanian.harga_komoditas WHERE nm_komoditas = %s GROUP BY tanggal"
    cursor.execute(sql)

    dateStart = datetime.strptime(dateRange[0], '%Y/%m/%d')
    dateEnd = datetime.strptime(dateRange[1], '%Y/%m/%d')
    dateStartStr = dateRange[0].replace('/', '-')
    dateEndStr = dateRange[1].replace('/', '-')

    df = None

    def dateParse(x): return datetime.strptime(x, "%Y-%m-%d")
    df = pd.read_sql_query()
    return df


@app.route('/getdata', methods=['GET'])
def getDataframe():
    msg = 'Hello world!'
    print(msg)  
    return msg

@app.route('/prediksi', methods=['GET'])
def predict(comodity, dateRange, request):
    return


@app.route('/train-data', methods=['POST'])
def train(comodity, dateRange):
    try:
        comodity = ''
        print('Latih data!')
    except:
        print('Data tidak dapat dilatih')
    return comodity


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
