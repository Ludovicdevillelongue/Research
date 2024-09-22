import torch
import numpy as np
from Autoencoder_Asset_Pricing.data_management.get_data import ModelData
from Autoencoder_Asset_Pricing.lstm_autoencoder.simple_conditional_autoencoders import ParamsIO, CA0, CA1, CA2, CA3
from Autoencoder_Asset_Pricing.lstm_autoencoder.lstm_conditional_autoencoders import ParamsIO_LSTM, CA0_LSTM, CA1_LSTM, \
    CA2_LSTM, CA3_LSTM
from Autoencoder_Asset_Pricing.lstm_autoencoder.af_lstm_conditional_autoencoders import ParamsIO_AF_LSTM, CA0_AF_LSTM, \
    CA1_AF_LSTM, \
    CA2_AF_LSTM, CA3_AF_LSTM
import os
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import math
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import pandas as pd
from functools import reduce
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class EarlyStopping:
    """
    This class allow to use early stopping in a model that is not hyper-optimized
    Stop the training as soon as the validation error reaches the minimum
    Took on stackoverflow
    """

    def __init__(
            self,
            tolerance: int = None,
            min_delta: int = 10):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class ModelCA:
    """
    This class allows to:
    _Load the data created and transformed
    _Instantiate a model defined above
    _Define the main args (configuration of hyper parameter, optimizer, criterion)
    _Define the main methods (training, validation, test) for the models defined
    _Define the main metrics (loss, r2)
    """

    def __init__(
            self,
            encoder_type: str = None,
            model_type: str = None,
            K_factor: int = None,
            max_epochs: int = None,
            tolerance_es: int = None
    ):
        self.encoder_type = encoder_type
        self.model_type = model_type
        self.K_factor = K_factor
        self.max_epochs = max_epochs
        self.tolerance_es = tolerance_es
        self.data_model = ModelData().load_args()
        self.config = None
        self.model = None
        self.criterion = None
        self.optimizer = None

    """
    -------------------------------------------
    ------------------Train---------------------
    -------------------------------------------
    """

    def set_model(self):
        shape_X_beta_train_i = self.data_model.X_1_train[0].shape
        shape_X_factor_train_i = self.data_model.X_2_train[0].shape
        if self.encoder_type == "Simple":
            params = ParamsIO(input_dim_beta=shape_X_beta_train_i[1], input_dim_factor=shape_X_factor_train_i[1],
                              output_dim_beta=self.K_factor, output_dim_factor=self.K_factor)
            if self.model_type == 'CA0':
                self.model = CA0(params)
            elif self.model_type == 'CA1':
                self.model = CA1(params, self.config)
            elif self.model_type == 'CA2':
                self.model = CA2(params, self.config)
            elif self.model_type == 'CA3':
                self.model = CA3(params, self.config)
            else:
                raise Exception(f'{self.model_type} has not been implemented')
        elif self.encoder_type == "LSTM":
            params = ParamsIO_LSTM(input_dim_beta=shape_X_beta_train_i[1], input_dim_factor=shape_X_factor_train_i[1],
                                   output_dim_beta=self.K_factor, output_dim_factor=self.K_factor)
            if self.model_type == 'CA0':
                self.model = CA0_LSTM(params)
            elif self.model_type == 'CA1':
                self.model = CA1_LSTM(params, self.config)
            elif self.model_type == 'CA2':
                self.model = CA2_LSTM(params, self.config)
            elif self.model_type == 'CA3':
                self.model = CA3_LSTM(params, self.config)
            else:
                raise Exception(f'{self.model_type} has not been implemented')
        elif self.encoder_type == "AF LSTM":
            params = ParamsIO_AF_LSTM(input_dim_beta=shape_X_beta_train_i[1],
                                      input_dim_factor=shape_X_factor_train_i[1],
                                      output_dim_beta=self.K_factor, output_dim_factor=self.K_factor)
            if self.model_type == 'CA0':
                self.model = CA0_AF_LSTM(params)
            elif self.model_type == 'CA1':
                self.model = CA1_AF_LSTM(params, self.config)
            elif self.model_type == 'CA2':
                self.model = CA2_AF_LSTM(params, self.config)
            elif self.model_type == 'CA3':
                self.model = CA3_AF_LSTM(params, self.config)
            else:
                raise Exception(f'{self.model_type} has not been implemented')

        """ 
        #Useless with Natixis laptop
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        self.model.to(device)
        """

    def instantiate_args(self, config: dict):
        self.config = config
        self.set_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.criterion = torch.nn.MSELoss()

    def train_one_epoch(self, X_1, X_2, Y):
        X_beta = X_1
        X_factor = X_2
        Y = Y
        X_beta_i = X_beta[0, :, :]
        X_factor_i = X_factor[0, :, :]
        Y_i = Y[0, :, :].T
        train_loss = 0.0
        self.model.train()  # puts the model in training mode (batch normalization and dropout are used)
        self.optimizer.zero_grad()
        outputs = self.model(X_beta, X_factor)  # y_pred[1][0] -> model.forward
        loss = self.criterion(outputs, Y_i)
        l1_parameters = torch.cat([parameter.view(-1) for parameter in self.model.parameters() if
                                   len(parameter.view(-1)) != self.K_factor])
        l1_parameters_ = torch.cat([parameter.view(-1) for parameter in self.model.parameters()])
        # print(f'just weights:{torch.norm(l1_parameters, 1)}')
        # print(f'weights + bias:{torch.norm(l1_parameters_, 1)}')
        l1_reg = self.config["l1"] * torch.norm(l1_parameters, 1)
        loss += l1_reg
        loss.backward()
        self.optimizer.step()
        return outputs, loss.item()

    @staticmethod
    def compute_r2_steps(encoder_type: str, numerator: list, denumerator: list, outputs: torch.Tensor,
                         Y_valid_: torch.Tensor):
        Y_valid_ = torch.reshape(Y_valid_, (-1,))
        predicted, _ = torch.max(outputs.data[:, 0], 0)
        numerator.append(((Y_valid_ - predicted) ** 2).sum().item())
        denumerator.append((Y_valid_ ** 2).sum().item())
        return numerator, denumerator

    def plot_loss_train_valid_test(self, loss, set):
        pass

    def compute_prediction(self, X_beta: torch.Tensor, X_factor: torch.Tensor, Y_valid: torch.Tensor):
        sample_loss = 0.0
        numerator_r2, denumerator_r2 = [], []
        self.model.eval()
        # puts the model in testing mode
        Y_valid_i = Y_valid[0, :, :].T
        X_beta_i = X_beta[0, :, :]
        X_factor_i = X_factor[0, :, :]

        outputs = self.model(X_beta, X_factor)
        loss = self.criterion(outputs, Y_valid)
        sample_loss += loss.item()
        numerator_r2, denumerator_r2 = self.compute_r2_steps(self.encoder_type, numerator_r2, denumerator_r2,
                                                             outputs, Y_valid)
        r2_total = 1 - np.sum(numerator_r2) / np.sum(denumerator_r2)
        sample_loss /= (len(Y_valid_i.T) * len(Y_valid))  # nb of stocks predicted * nb of days in the sample
        return sample_loss, r2_total

    def validate_one_epoch(self):
        X_beta = self.data_model.X_1_valid
        X_factor = self.data_model.X_2_valid
        Y_valid = self.data_model.Y_valid
        valid_loss, r2_total = self.compute_prediction(X_beta, X_factor, Y_valid)
        r2_total = r2_total if r2_total > 0 else 0  # in order to maximize r2 (do not want to maximize negative value)
        return valid_loss, r2_total

    def train_model(self, X_1, X_2, Y, config: dict, hyperopt: bool, set: str):

        if not hyperopt: early_stopping = EarlyStopping(tolerance=self.tolerance_es, min_delta=10)
        self.instantiate_args(config)
        train_loss, validation_loss, test_loss = [], [], []
        for t in range(self.max_epochs):
            # training
            if set == "training_validation":
                outputs, epoch_train_loss = self.train_one_epoch(X_1, X_2, Y)
                train_loss.append(epoch_train_loss)
                # validation
                # Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward()
                with torch.no_grad():
                    epoch_validate_loss, epoch_validate_r2 = self.validate_one_epoch()
                    validation_loss.append(epoch_validate_loss)
                if t % 10 == 0 and t != 0:
                    print("Epoch: {} Train Loss: {} Val Loss: {} Val R2: {}\\".format(
                        t, epoch_train_loss, epoch_validate_loss, epoch_validate_r2))

            # testing
            else:
                outputs, epoch_test_loss = self.train_one_epoch(X_1, X_2, Y)
                test_loss.append(epoch_test_loss)
                if t % 10 == 0 and t != 0:
                    print("Epoch: {} Test Loss: {}".format(
                        t, epoch_test_loss))

        if set == "training_validation":
            self.plot_loss_train_valid_test(train_loss, "{} Conditional Autoencoder Training Loss".format(self.encoder_type))
            self.plot_loss_train_valid_test(validation_loss, "{} Conditional Autoencoder Validation Loss".format(self.encoder_type))
        else:
            self.plot_loss_train_valid_test(test_loss, "{} Conditional Autoencoder Testing Loss".format(self.encoder_type))
        return outputs

    def plot_predictions(self, actual, pred, set):
        pass

    def plot_residuals(self, residuals, path):
        pass

    # def plot_extreme_values(self, df_dates_values,set, path):
    #
    #     for res_ticker in range(0,len(df_dates_values.columns)):
    #         spacing = 2
    #         fig, (ax1, ax2) = plt.subplots(2, 1)
    #         ax1.scatter(df_dates_values.index, df_dates_values.iloc[:,res_ticker])
    #         ax1.set_xlabel('Time')
    #         ax1.set_ylabel('Errors')
    #         ax1.set_title(self.data_model.tickers[res_ticker] +" "+ path)
    #
    #         if set=="Train":
    #             ax2.plot(((self.data_model.initial_data[self.data_model.tickers[res_ticker]]).index)
    #                      [0:len(self.data_model.X_1_train)],
    #                      ((self.data_model.initial_data[self.data_model.tickers[res_ticker]])["log_ret"])
    #                      [0:len(self.data_model.X_1_train)])
    #             ax2.set_xlabel('Time')
    #             ax2.set_ylabel('Log Returns')
    #             ax2.set_title(self.data_model.tickers[res_ticker] + " Log Returns Evolution on Train Set")
    #         elif set == "Test":
    #             ax2.plot(((self.data_model.initial_data[self.data_model.tickers[res_ticker]]).index)
    #                      [len(self.data_model.X_1_train) + len(self.data_model.X_1_valid) + 1:],
    #                      ((self.data_model.initial_data[self.data_model.tickers[res_ticker]])["log_ret"])
    #                      [len(self.data_model.X_1_train) + len(self.data_model.X_1_valid) + 1:])
    #             ax2.set_xlabel('Time')
    #             ax2.set_ylabel('Log Returns')
    #             ax2.set_title(self.data_model.tickers[res_ticker] + " Log Returns Evolution on Test Set")
    #
    #
    #         # for each label in the list of xticklabels
    #         # (we skip every n'th label in this list using [::n]):
    #         # set to not visible
    #         for label in ax1.xaxis.get_ticklabels()[::spacing]:
    #             label.set_visible(False)
    #         for label in ax2.xaxis.get_ticklabels()[::spacing]:
    #             label.set_visible(False)
    #
    #         plt.subplots_adjust(left=0.1,
    #                             bottom=0.1,
    #                             right=0.9,
    #                             top=0.9,
    #                             wspace=0.1,
    #                             hspace=0.9)
    #         plt.savefig("results_1y/{}/{} {}.png".format(self.model_type,self.data_model.tickers[res_ticker], path), bbox_inches="tight")

    def plot_extreme_values(self, df_dates_values, set, path):
        for res_ticker in range(len(df_dates_values.columns)):

            fig = go.Figure()

            # add log returns trace
            if set == "Train":
                log_returns = self.data_model.initial_data[self.data_model.tickers[res_ticker]]["log_ret"][
                              0:len(self.data_model.X_1_train)]
                dates = self.data_model.initial_data[self.data_model.tickers[res_ticker]].index[
                        0:len(self.data_model.X_1_train)]
            elif set == "Test":
                log_returns = self.data_model.initial_data[self.data_model.tickers[res_ticker]]["log_ret"][
                              len(self.data_model.X_1_train) + len(self.data_model.X_1_valid) + 1:]
                dates = self.data_model.initial_data[self.data_model.tickers[res_ticker]].index[
                        len(self.data_model.X_1_train) + len(self.data_model.X_1_valid) + 1:]

            fig.add_trace(go.Scatter(x=dates, y=(log_returns/100), name="Log Returns", line=dict(width=2, color='black')))

            # add errors trace on secondary y-axis
            fig.add_trace(go.Scatter(x=df_dates_values.index, y=df_dates_values.iloc[:, res_ticker],mode="markers",
                                     name="Errors", yaxis="y2",marker=dict(size=7, color='red')))

            # create axis objects
            fig.update_layout(
                title={'text': self.data_model.tickers[res_ticker] + " " + path, 'y': 0.9, 'x': 0.5,
                       'xanchor': 'center', 'yanchor': 'top'},
                xaxis=dict(title="Dates", showline=True, showgrid=True, showticklabels=True, linecolor='grey',
                           linewidth=2, ticks='outside', tickfont=dict(family='Arial', size=12, color='black'),
                           gridcolor='LightGrey'),
                yaxis=dict(title="Log Returns", showgrid=True, side='left', titlefont=dict(color='black'),
                           tickfont=dict(color='black'), gridcolor='LightGrey'),
                yaxis2=dict(title="Errors", titlefont=dict(color='black'),
                            tickfont=dict(color='black'),
                            anchor="x", overlaying="y", side="right", showgrid=False),
                legend=dict(x=1.05, y=1, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.5)'),
                margin=dict(autoexpand=True, l=100, r=100, t=110, ),
                plot_bgcolor='white')

            fig.show()

            # save to HTML
            fig.write_html(
                "results_1y/{}/{} {}.html".format(self.model_type, self.data_model.tickers[res_ticker], path))

    def predictions(self):
        if self.encoder_type == "Simple":
            trainPredict = self.train_model(self.data_model.X_1_train, self.data_model.X_2_train,
                                            self.data_model.Y_train
                                            , config={'l1': 0.01, 'lr': 0.001, 'rate_dropout': 0}, hyperopt=False,
                                            set="training_validation")
            testPredict = self.train_model(self.data_model.X_1_test, self.data_model.X_2_test, self.data_model.Y_test,
                                           config={'l1': 0.01, 'lr': 0.001, 'rate_dropout': 0}, hyperopt=False,
                                           set="testing")
        elif self.encoder_type == "LSTM":
            trainPredict = self.train_model(self.data_model.X_1_train, self.data_model.X_2_train,
                                            self.data_model.Y_train
                                            , config={'l1': 0.0016, 'lr': 0.001, 'rate_dropout': 0}, hyperopt=False,
                                            set="training_validation")
            testPredict = self.train_model(self.data_model.X_1_test, self.data_model.X_2_test, self.data_model.Y_test,
                                           config={'l1': 0.0016, 'lr': 0.001, 'rate_dropout': 0}, hyperopt=False,
                                           set="testing")
        elif self.encoder_type == "AF LSTM":
            trainPredict = self.train_model(self.data_model.X_1_train, self.data_model.X_2_train,
                                            self.data_model.Y_train
                                            , config={'l1': 0.0000012, 'lr': 0.001, 'rate_dropout': 0}, hyperopt=False,
                                            set="training_validation")
            testPredict = self.train_model(self.data_model.X_1_test, self.data_model.X_2_test, self.data_model.Y_test,
                                           config={'l1': 0.0000012, 'lr': 0.001, 'rate_dropout': 0}, hyperopt=False,
                                           set="testing")

        # new tensor with computed error function
        err_train = torch.special.erf(trainPredict)
        err_test = torch.special.erf(testPredict)

        # convert tensor to array
        trainPredict = self.data_model.scaler_Y.inverse_transform(
            trainPredict[:, :, 0].detach().numpy())
        testPredict = self.data_model.scaler_Y.inverse_transform(
            testPredict[:, :, 0].detach().numpy())

        Y_train_actual = self.data_model.scaler_Y.inverse_transform(
            self.data_model.Y_train[:, 0, :])
        Y_test_actual = self.data_model.scaler_Y.inverse_transform(
            self.data_model.Y_test[:, 0, :])

        # plot actual returns vs predictions
        self.plot_predictions(Y_train_actual, trainPredict,
                              "{} Conditional Autoencoder Predictions on train set".format(self.encoder_type))
        self.plot_predictions(Y_test_actual, testPredict,
                              "{} Conditional Autoencoder Predictions on test set".format(self.encoder_type))
        return err_train, err_test, trainPredict, Y_train_actual, testPredict, Y_test_actual

    def anomaly_detection(self):
        err_train, err_test, trainPredict, Y_train_actual, testPredict, Y_test_actual = self.predictions()

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(Y_train_actual, trainPredict))
        print('Train Score: %.6f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(Y_test_actual, testPredict))
        print('Test Score: %.6f RMSE' % (testScore))

        # calculate residuals
        train_residuals = [Y_train_actual[i] - trainPredict[i] for i in range(len(trainPredict))]

        test_residuals = [Y_test_actual[i] - testPredict[i] for i in range(len(testPredict))]

        # plot residuals
        self.plot_residuals(train_residuals,
                            "{} Conditional Autoencoder Residuals Evolution on Train Set".format(self.encoder_type))
        self.plot_residuals(test_residuals, "{} Conditional Autoencoder Residuals Evolution on Test Set".format(self.encoder_type))

        # find dates where residuals are too high
        def extreme_values(residuals, set, quant):
            anomaly_dates = []
            anomaly_values = []
            anomaly_dates_values = []
            residuals = pd.DataFrame(residuals)
            for i in range(0,2):
                anomaly_dates_ticker = []
                anomaly_values_ticker = []
                for j, res_ticker in enumerate(list(residuals.iloc[:, i])):
                    if abs(res_ticker) > np.quantile((list(map(abs, residuals.iloc[:, i]))), quant):
                        if set == "train":
                            anomaly_dates_ticker.append((self.data_model.initial_data["BTC"].index.values)[j + 1])
                            anomaly_values_ticker.append(residuals.iloc[j, i])
                        elif set == "test":
                            anomaly_dates_ticker.append((self.data_model.initial_data["BTC"].index.values)
                                                        [j + len(self.data_model.X_1_train) + len(
                                    self.data_model.X_1_valid) + 1])
                            anomaly_values_ticker.append(residuals.iloc[j, i])
                # create dataframe to join dates and extreme values for one ticker
                anomaly_values_dates_ticker = pd.DataFrame([anomaly_dates_ticker, anomaly_values_ticker]).T
                anomaly_values_dates_ticker.columns = ["Dates", self.data_model.tickers[i]]
                # add dataframe for each ticker to a list
                anomaly_dates_values.append(anomaly_values_dates_ticker)
            # create dataframe of dates and extreme values for all tickers
            df_anomaly_dates_values = reduce(lambda left, right: pd.merge(left, right, on=["Dates"], how="outer"),
                                             anomaly_dates_values)
            df_anomaly_dates_values = df_anomaly_dates_values.sort_values(by='Dates')
            df_anomaly_dates_values.set_index("Dates", inplace=True)
            return df_anomaly_dates_values

        anomaly_train_5_perc = extreme_values(train_residuals, "train", 0.95)
        anomaly_test_5_perc = extreme_values(test_residuals, "test", 0.95)

        anomaly_train_5_perc.to_csv(
            "results_1y/{}/{} Conditional Autoencoder 5 Percent Anomalies per Ticker on Train Set.csv".format(self.model_type,
                                                                                                  self.encoder_type))
        anomaly_test_5_perc.to_csv(
            "results_1y/{}/{} Conditional Autoencoder 5 Percent Anomalies per Ticker on Test Set.csv".format(self.model_type,
                                                                                                 self.encoder_type))
        self.plot_extreme_values(anomaly_train_5_perc, "Train",
                                 "{} Conditional Autoencoder 5 Percent Extreme Values vs Log Returns Evolution on Train Set".format(
                                     self.encoder_type))
        self.plot_extreme_values(anomaly_test_5_perc, "Test",
                                 "{} Conditional Autoencoder 5 Percent Extreme Values vs Log Returns Evolution on Test Set".format(
                                     self.encoder_type))