from flask import Flask
from datetime import datetime
import pandas as pd
import pymysql

app = Flask(__name__)

# Configuration to MySQL
CONNECTION = pymysql.connect(
    HOST='localhost',
    USER='root',
    PASSWORD='',
    DATABASE='pertanian'
)
cursor = CONNECTION.cursor()

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


@app.route('/prediksi', methods=['POST'])
def predict(comodity, dateRange, request):
    if request.method == 'GET':
        comodity = ''
        dateRange = []
        percentage = 0

        df = None

        dateAwal = datetime.strptime(dateRange[0], '%Y/%m/%d').timestamp()
        dateAkhir = datetime.strptime(dateRange[1], '%Y/%m/%d').timestamp()

    elif request.method == 'POST':
        request = ''

    if comodity == '' or len(dateRange) == 0 or percentage == 0:
        print('Silahkan isi terlebih dahulu')

    return df


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
