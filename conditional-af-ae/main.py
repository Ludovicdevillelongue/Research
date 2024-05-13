from datetime import datetime
from data_management.get_data import DataConstructor, ModelData
from models.training import ModelCA


if __name__ == '__main__':

    #list of crypto to retrieve
    tickers=["BTC","ETH"]

    #date from which to retrieve data and frequency
    start_date = datetime.strptime("2021-08-16 00:00:00", '%Y-%m-%d %H:%M:%S').date()
    frequency = "hourly"

    # identification for simons_backend_sdk
    client_id = "ludovic"
    client_secret = "_nd_xFCeZVEFWrTYg6q5tAod0k2J05DlW7JZCpTLLXOsbdu8"


    # #To debug a model

    #get crypto and factor data
    data=DataConstructor(tickers,start_date,client_id,client_secret,frequency)
    #Instantiation and computation of the data
    ModelData(data).compute()
    model_simple = ModelCA(encoder_type="Simple", model_type='CA3', K_factor=1, max_epochs=200, tolerance_es=10)
    model_simple.anomaly_detection()
    model_LSTM = ModelCA(encoder_type="LSTM", model_type='CA3', K_factor=1, max_epochs=200, tolerance_es=10)
    model_LSTM.anomaly_detection()
    model_AF_LSTM = ModelCA(encoder_type="AF LSTM", model_type='CA3', K_factor=1, max_epochs=200, tolerance_es=10)
    model_AF_LSTM.anomaly_detection()


