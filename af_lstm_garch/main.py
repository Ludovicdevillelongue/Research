import datetime
import math
import os

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from modules import AF_LSTM
from modules import GARCH
from modules import CustomLSTM
from sklearn.preprocessing import MinMaxScaler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':

    # Data import
    lookback = 5
    start = datetime.datetime(2014, 9, 17)
    end = datetime.datetime(2022, 6, 20)
    crypto_symbols=["BTC","ETH"]
    RMSE_cumul=[]

    for crypto, crypto_symbol in enumerate(crypto_symbols):
        ticker_yahoo = yf.Ticker("{}-USD".format(crypto_symbol))
        data = ticker_yahoo.history(start=start, end=end)
        df = pd.DataFrame(data["Close"], columns=["Close"])
        garch_model = GARCH(df)
        log_ret = garch_model.compute_log_ret()
        forecast_log_returns = garch_model.forecast_log_ret(log_ret,crypto_symbol)
        dict_train_test = garch_model.dataset_creation(lookback)


        def loss_function(model, optimizer, loss_fn, dict_train_test, num_epochs=1000):
            global y_pred_train, y_pred_test
            # Extracting training and test data from the dictionary
            x_train = dict_train_test["x_train"]["x_train_data"]
            y_train = dict_train_test["y_train"]["y_train_data"]
            x_test = dict_train_test["x_test"]["x_test_data"]
            y_test = dict_train_test["y_test"]["y_test_data"]

            train_loss_history = []
            test_loss_history = []

            for epoch in range(num_epochs):
                model.train()
                # Forward pass
                y_pred_train = model(x_train)
                loss = loss_fn(y_pred_train[1][0].unsqueeze(1)[:,:,0], y_train)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                average_train_loss = loss.item() / len(x_train)
                train_loss_history.append(average_train_loss)

                # Evaluation on test set
                model.eval()
                with torch.no_grad():
                    y_pred_test = model(x_test)
                    loss = loss_fn(y_pred_test[1][0].unsqueeze(1)[:,:,0], y_test)
                average_test_loss = loss.item() / len(x_test)
                test_loss_history.append(average_test_loss)
                if epoch % 10 == 0 and epoch != 0:
                    print(f'Epoch {epoch}, Train Loss: {average_train_loss}, Test Loss: {average_test_loss}')

            # Plotting the losses
            plt.plot(train_loss_history)
            plt.plot(test_loss_history)
            return y_pred_train, y_pred_test


        def predictions(model, optimiser, dict_train_test):
            # make predictions
            trainPredict, testPredict = loss_function(model, optimiser,torch.nn.MSELoss(), dict_train_test)
            # inverse scaling for forecast
            trainPredict = dict_train_test["y_train"]["y_train_scaler"].inverse_transform(
                trainPredict[1][0].detach().numpy())[:, 0]
            testPredict = dict_train_test["y_test"]["y_test_scaler"].inverse_transform(
                testPredict[1][0].detach().numpy())[:, 0]

            # inverse scaling for actual
            y_train_actual = dict_train_test["y_train"]["y_train_scaler"].inverse_transform(
                dict_train_test["y_train"]["y_train_data"])[:, 0]
            y_test_actual = dict_train_test["y_test"]["y_test_scaler"].inverse_transform(
                dict_train_test["y_test"]["y_test_data"])[:, 0]

            # calculate root mean squared error
            trainScore = math.sqrt(mean_squared_error(y_train_actual, trainPredict))
            print('Train Score: %.6f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(y_test_actual, testPredict))
            print('Test Score: %.6f RMSE' % (testScore))

            return y_train_actual,trainPredict,y_test_actual, testPredict, trainScore, testScore


        # LSTM Creation
        plt.figure()
        model = CustomLSTM(input_size=2, hidden_size=64)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
        # scheduler torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min')
        print(model)
        print(len(list(model.parameters())))
        for i in range(len(list(model.parameters()))):
            print(list(model.parameters())[i].size())
        predictions_lstm = predictions(model, optimiser, dict_train_test)


        # AF LSTM creation
        model_2 = AF_LSTM(input_size=2, hidden_size=64, max_seqlen=1000, output_size=64)
        optimiser_2 = torch.optim.Adam(model_2.parameters(), lr=0.001)
        print(model_2)
        print(len(list(model_2.parameters())))
        for i in range(len(list(model_2.parameters()))):
            print(list(model_2.parameters())[i].size())
        predictions_af_lstm = predictions(model_2, optimiser_2, dict_train_test)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('{} MSE loss'.format(crypto_symbol))
        plt.legend(["LSTM train", "LSTM test", "AF LSTM train", "AF LSTM test"])
        plt.savefig("results_10_tickers/{} Loss Function.png".format(crypto_symbol))


        RMSE_cumul.append([predictions_lstm[4],predictions_lstm[5], predictions_af_lstm[4],predictions_af_lstm[5]])





        # Predictions plots
        def plot_predictions(actual, pred):
            plt.plot(actual, label="volatility")
            plt.plot(pred, label="volatility")
            plt.title('{} Volatility Prediction'.format(crypto_symbol))
            plt.xlabel('Time')
            plt.ylabel('{} Volatility'.format(crypto_symbol))
            plt.legend(loc='upper right')

        plt.figure()
        plot_predictions(predictions_lstm[2], predictions_lstm[3])
        plt.legend(["LSTM actual volatility on test set", "LSTM predicted volatility on test set"],loc='upper right')
        plt.savefig("results_10_tickers/{} Volatility LSTM test set.png".format(crypto_symbol))


        plt.figure()
        plot_predictions(predictions_af_lstm[2], predictions_af_lstm[3])
        plt.legend(["AF LSTM actual volatility on test set", "AF LSTM predicted volatility on test set"],loc='upper right')
        plt.savefig("results_10_tickers/{} Volatility AF LSTM test set.png".format(crypto_symbol))

        plt.figure()
        plot_predictions(predictions_lstm[0], predictions_lstm[1])
        plt.legend(["LSTM actual volatility on train set", "LSTM predicted volatility on train set"],loc='upper right')
        plt.savefig("results_10_tickers/{} Volatility LSTM train set.png".format(crypto_symbol))


        plt.figure()
        plot_predictions(predictions_af_lstm[0], predictions_af_lstm[1])
        plt.legend(["AF LSTM actual volatility on train set", "AF LSTM predicted volatility on train set"],loc='upper right')
        plt.savefig("results_10_tickers/{} Volatility AF LSTM train set.png".format(crypto_symbol))



    df_RMSE=pd.DataFrame(RMSE_cumul,columns=["LSTM Train Score","LSTM Test Score","AF LSTM Train Score",
                                             "AF LSTM Test Score"],index=crypto_symbols)
    df_RMSE.to_csv("RMSE_per_ticker.csv")


