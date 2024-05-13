
import json
from datetime import datetime, date, timedelta
import pandas as pd
import sys
import os
from matplotlib import pyplot as plt
import time
from Autoencoder_Asset_Pricing.data_management.quote_connect import get_category
import pandas_ta as ta
from typing import List, Dict, Tuple
import pickle
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import fathon
from fathon import fathonUtils as fu




"""
---------------------------
Data Download and Treatment
---------------------------
"""
class DataConstructor:
    def __init__(self,tickers,start_date,client_id,client_secret,frequency):
        self.tickers=tickers
        self.start_date=start_date
        self.client_id=client_id
        self.client_secret=client_secret
        self.frequency=frequency
        self.create_factor_features()

    def compute_log_ret(self,data):
        data['log_ret'] = np.log(data['close']/data['close'].shift(1))*100
        return data

    def get_hurst_exponent(self,time_series,window):
        # zero-mean cumulative sum
        time_series= fu.toAggregated(time_series)

        # initialize ht object
        pyht = fathon.HT(time_series)

        # compute time-dependent Hurst exponent
        ht = np.array(pyht.computeHt(window, mfdfaPolOrd=1, polOrd=1))

        #fill missing values
        nparr = np.full((window-1), np.nan)
        ht_tot=np.append(nparr, ht.T)
        return ht_tot


    def compute_talib_fields(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        As there is not enough features (characteristics) from the bloomberg request, used the talib package which
        perform technical analysis of financial market data and help identifying different patterns that stocks follow
        """

        for ticker in ['BTC', 'ETH']:
            inputs = {
                'open': data[ticker]['open'],
                'high': data[ticker]['high'],
                'low': data[ticker]['low'],
                'close': data[ticker]['close'],
                'volume': data[ticker]['volume']
            }
            #factors
            data[ticker]['sma']=ta.sma(data[ticker]['close'],24)
            data[ticker]['dema']=ta.dema(data[ticker]['close'],12)
            data[ticker]['cci'] = ta.cci(data[ticker]['high'],data[ticker]['low'],data[ticker]['close'],24)
            data[ticker]['ad'] = ta.ad(data[ticker]['high'],data[ticker]['low'],data[ticker]['close'],
                                        data[ticker]['volume'],data[ticker]['open'])
            data[ticker]['atr']=ta.atr(data[ticker]['high'],data[ticker]['low'],data[ticker]['close'], 24)

            #hurst
            data[ticker] = self.compute_log_ret(data[ticker])
            window=24
            data[ticker]["hurst_exponent"]=self.get_hurst_exponent(data[ticker]['log_ret'].values,window)
            data[ticker]["returns"]=data[ticker]['close'].pct_change()
            data[ticker]=data[ticker].dropna()
        return data, ['sma','dema','cci','ad','atr','hurst_exponent','returns']

    def save_data(self,dict_crypto):
        with open('data_1y/data.pickle', 'wb') as handle:
            pickle.dump(dict_crypto, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        with open('data_1y/data.pickle', 'rb') as file:
            data = pickle.load(file)
        return data

    def create_factor_features(self):
        try:
            self.data = self.load_data()
        except FileNotFoundError:
            self.data=self.call_simons_backend_sdk()
            self.save_data(self.data)

        self.data, self.field=self.compute_talib_fields(self.data)
        self.start = self.data[list(self.data.keys())[0]].index[-1]
        self.end = self.data[list(self.data.keys())[0]].index[0]
        self.rows = len(self.data[list(self.data.keys())[0]])
        self.cols = len(self.data[list(self.data.keys())[0]].T)
        self.keys = self.data.keys()


class ModelData:
    """
    This class allows to create the Xs data (input 1 being asset characteristics (in t-1) and input 2 being asset
    excess returns (in t))
    Then the Xs are split for the training sample, validation sample and testing sample
    Moreover the training, validation and testing set are scaled (Min Max scaler)
    And conversion to tensor
    """

    def __init__(
            self,
            data_base: DataConstructor = None,
            training_proportion: float = 0.7,
            validation_proportion: float = 0.15,
            testing_proportion: float = 0.15
    ):
        self.data_base = data_base
        self.training_proportion = training_proportion
        self.validation_proportion = validation_proportion
        self.testing_proportion = testing_proportion
        self.dict_keys={key: 0 for key in ['BTC', 'ETH']}

    def init_X_1(self):
        """
        From the dataframe, instantiation of array data that will serve in the beta network i.e characteristics
        """
        features = pd.concat(list(
            map(lambda index: pd.concat(
                list(map(lambda tick: self.data_base.data[tick].iloc[index], self.dict_keys)),
                axis=1, keys=self.dict_keys).T, range(self.data_base.rows-1))))  # [0,len(data)-1]
        features=features.iloc[:,8:-3]
        return np.array(features).reshape(self.data_base.rows-1, len(self.dict_keys), self.data_base.cols-11)


    def init_X_2(self):
        """
        From the dataframe, instantiation of array data that will serve in the factor network i.e heurst exponent
        """
        hurst_exponent = pd.concat(
            list(map(lambda tick: self.data_base.data[tick]['hurst_exponent'], self.dict_keys)),
            axis=1, keys=self.dict_keys) .iloc[1:]  # [1,len(data)]
        return np.array(hurst_exponent).reshape(self.data_base.rows-1, 1, len(self.dict_keys))

    def init_Y(self):
        returns = pd.concat(
            list(map(lambda tick: self.data_base.data[tick]['returns'], self.dict_keys)),
            axis=1, keys=self.dict_keys).iloc[1:]  # [1,len(data)]
        return np.array(returns).reshape(self.data_base.rows-1, 1, len(self.dict_keys))



    def init_scaler(self):
        """
        Type of scaler to rank-normalize asset characteristics into the interval (-1,1)
        The rank normalization is done on each day for the asset characteristic of each tickers
        """
        return MinMaxScaler(feature_range=(-1, 1))

    def set_X(self, start: int, end: int, type: str):
        """
        Converting raw data array into scaled tensor sample (train, valid, test)
        """
        if type =='training':
            X_1 = np.stack(list(map(lambda index: self.scaler_X_1.fit_transform(self.X_1[start:end,:,:][index]),
                                    range(len(self.X_1[start:end])))))
        else:
            X_1 = np.stack(list(map(lambda index: self.scaler_X_1.transform(self.X_1[start:end][index]),
                                    range(len(self.X_1[start:end])))))
        X_2 = self.X_2[start:end]
        return torch.tensor(X_1, dtype=torch.float32), torch.tensor(X_2, dtype=torch.float32)

    def set_Y(self, start: int, end: int, type: str):
        if type =='training':
            Y = self.scaler_Y.fit_transform(self.Y[start:end].reshape(-1, self.Y[start:end].shape[-1]))\
                .reshape(self.Y[start:end].shape)
        else:
            Y = self.scaler_Y.transform(self.Y[start:end].reshape(-1, self.Y[start:end].shape[-1])) \
                .reshape(self.Y[start:end].shape)
        return torch.tensor(Y, dtype=torch.float32)



    def init_train_data(self) -> int:
        """
        Instantiation of the training data
        """
        self.scaler_X_1 = self.init_scaler()
        self.scaler_Y=self.init_scaler()
        start = 0
        end = int(self.data_base.rows * self.training_proportion)
        self.X_1_train, self.X_2_train = self.set_X(start, end, 'training')
        self.Y_train=self.set_Y(start, end, 'training')
        return end

    def init_valid_data(self) -> int:
        """
        Instantiation of the validation data
        """
        start = self.init_train_data()
        end = start + int(self.data_base.rows * self.validation_proportion)
        self.X_1_valid, self.X_2_valid = self.set_X(start, end, 'validation')
        self.Y_valid=self.set_Y(start, end, 'validation')
        return end

    def init_test_data(self):
        """
        Instantiation of the testing data
        """
        start = self.init_valid_data()
        end = start + int(self.data_base.rows * self.testing_proportion)
        self.X_1_test, self.X_2_test = self.set_X(start, end, 'testing')
        self.Y_test=self.set_Y(start, end, 'testing')

    def compute(self):
        """
        Main method of the class, allowing to perform the steps and saving the arguments because the computation is long
        """
        assert (0 <= self.training_proportion <= 1)
        assert (0 <= self.validation_proportion <= 1)
        assert (0 <= self.testing_proportion <= 1)
        assert (self.training_proportion + self.validation_proportion + self.testing_proportion == 1)
        self.X_1 = self.init_X_1()
        self.X_2 = self.init_X_2()
        self.Y=self.init_Y()
        assert (self.X_1.shape[0] == self.X_2.shape[0])
        assert (self.X_1.shape[1] == self.X_2.shape[2])
        self.init_train_data()
        self.init_valid_data()
        self.init_test_data()
        self.save_args()

    def save_args(self):
        """
        As the main method of the class is long in debug mode, saving the data to reload after
        """
        args = {'X_1_train': self.X_1_train,'X_1_scaler':self.scaler_X_1,
                'X_2_train': self.X_2_train, 'Y_train': self.Y_train,
                'Y_train_scaler':self.scaler_Y,'X_1_valid': self.X_1_valid,
                'X_2_valid': self.X_2_valid, 'Y_valid' : self.Y_valid,
                'X_1_test': self.X_1_test, 'X_2_test': self.X_2_test, 'Y_test' : self.Y_test,
                'fields': self.data_base.field, 'tickers':list(self.data_base.tickers),
                'initial_data':self.data_base.data}
        with open('data_1y/data_model.pkl', "wb") as file:
            pickle.dump(args, file)

    def load_args(self):
        with open('data_1y/data_model.pkl', "rb") as file:
            args = pickle.load(file)
        self.X_1_train = args['X_1_train']
        self.X_2_train = args['X_2_train']
        self.scaler_X_1=args['X_1_scaler']
        self.Y_train = args['Y_train']
        self.scaler_Y = args['Y_train_scaler']
        self.X_1_valid = args['X_1_valid']
        self.X_2_valid = args['X_2_valid']
        self.Y_valid = args['Y_valid']
        self.X_1_test = args['X_1_test']
        self.X_2_test = args['X_2_test']
        self.Y_test = args['Y_test']
        self.fields = args['fields']
        self.tickers=args['tickers']
        self.initial_data=args['initial_data']
        return self