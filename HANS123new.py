import pandas as pd
import numpy as np
import quantstats as qs
from scipy import stats
import datetime
from datetime import datetime, date, timedelta, time
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

class Hans123:
    def __init__(self, initcap=None):
        self.take_profit = None
        self.stop_loss = None
        self.initcap = initcap
        self.rr_ratio = 3   #risk reward ratio 1:3






    def preprocess_data(self, data, window_size):
        X, y = [], []
        for i in range(len(X) - window_size):
            v = data[i: (i + window_size)].values
            X.append(v)
            y.append(data[i+window_size])
        return np.array(X), np.array(y)
        print('hello1')


    def build_train_lstm_model(self, data, window_size, epochs=100):
        # # create dataset
        # X, y = preprocess_data(data, window_size)
        # X = X.reshape(X.shape[0], X.shape[1])


        #preprocess data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[['Open', 'Close', 'High', 'Low']])
        print('hello2')

        #create dataset
        X, y = self.preprocess_data(data_scaled, window_size)
        print('hello3')

        # train-test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[0:train_size], X[train_size:len(X)]
        y_train, y_test = y[0:train_size], y[train_size:len(y)]
        print('hello4')

        # LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.add(Dropout(0.2))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=epochs, batch_size=32)
        print('hello5')

        # Generate predictions for next day's market price
        last_30_days = data_scaled[-30:]
        last_30_days = last_30_days.reshape((1, last_30_days.shape[0], last_30_days.shape[1]))
        predicted_price = model.predict(last_30_days)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]
        print('hello6')
        return predicted_price

    def trading_strategy(self, data):
        # set initial values
        capital = self.initcap
        take_profit = 0
        stop_loss = 0
        position_size = capital * 0.01
        start_time = None
        end_time = None
        high = 0
        low = np.inf
        entry_price = None

        # wait for 30 mins and record high and low prices
        for i in range(len(data)):
            current_time = datetime.strptime(data['Dates'].iloc[i], "%Y.%m.%d %H:%M")
            if current_time.time() >= time(hour=9):
                start_time = current_time
                end_time = start_time + timedelta(minutes=30)
                print(start_time)
                print(end_time)
                print('12')
                break

        for i in range(len(data)):
            current_time = datetime.strptime(data['Dates'].iloc[i], "%Y.%m.%d %H:%M")

            if current_time >= start_time and current_time <= end_time:
                high = max(high, data['High'].iloc[i])
                low = min(low, data['Low'].iloc[i])


                print('13')

            elif current_time > end_time:
                print('14')
                break
        print(high)
        print(low)


        for i in range(len(data)):
            current_time = datetime.strptime(data['Dates'].iloc[i], "%Y.%m.%d %H:%M")

            if current_time > end_time:
                # execute trade if the market price rises above or below the high and low price
                if data['Close'].iloc[i] >= high:
                    entry_price = high
                    stop_loss = low
                    take_profit = entry_price + (self.rr_ratio * (entry_price - stop_loss))
                    predicted_take_profit = self.build_train_lstm_model(data.iloc[i:], window_size=60)
                    take_profit = max(take_profit, predicted_take_profit)  # use predicted take_profit if higher
                    profit = position_size * (take_profit - entry_price)
                    capital += profit
                    print('15')

                    break

                elif data['Close'].iloc[i] <= low:
                    entry_price = low
                    stop_loss = high
                    take_profit = entry_price - (self.rr_ratio * (entry_price - stop_loss))
                    predicted_take_profit = self.build_train_lstm_model(data.iloc[i:], window_size=60)
                    take_profit = min(take_profit, predicted_take_profit)  # use predicted take_profit if lower
                    loss = position_size * (stop_loss - entry_price)
                    capital -= loss
                    print('16')

                    break

                else:
                    print('17')

                    break

        return capital, entry_price, take_profit, stop_loss

    def run_strategy(self, EURUSDmins_data):
        #read csv
        data = pd.read_csv(r'C:\Users\jason.yam\EURUSDmins_data.csv')
        df = data.dropna()
        df['Dates'] = pd.to_datetime(df['Dates'], format="%d/%m/%Y %H:%M").dt.strftime("%Y.%m.%d %H:%M")
        # print(df)
        print('hello10')



        #execute trading strategy
        capital, entry_price, take_profit, stop_loss = self.trading_strategy(df)



        #statistics
        returns = (capital - self.initcap) / self.initcap
        daily_returns = returns / (len(data) / (24*60))
        mean = np.mean(data['Close'])
        std = np.std(data['Close'])
        corr = stats.pearsonr(data['Close'], data['Volume'])[0]
        sharpe_ratio = (daily_returns - 0.0377) / std        #assume risk-free rate = 3.77%

        print('Capital:', capital)
        print('Entry Price:', entry_price)
        print('Take Profit:', take_profit)
        print('Stop Loss:', stop_loss)
        print('Returns:', returns)
        print('Daily Returns:', daily_returns)
        print('Mean:', mean)
        print('Standard Deviation:', std)
        print('Correlation:', corr)
        print('Sharpe Ratio:', sharpe_ratio)

        return capital, entry_price, take_profit, stop_loss


if __name__ == '__main__':
    strategy = Hans123(initcap=10000)
    capital, entry_price, take_profit, stop_loss = strategy.run_strategy(r'C:\Users\jason.yam\EURUSDmins_data.csv')









    #     # Evaluate the model
    #     mse = model.evaluate(X_test, y_test, verbose=0)
    #     print('Mean Squared Error:', mse)
    #
    #     # Predictions
    #     train_predict = model.predict(X_train)
    #     test_predict = model.predict(X_test)
    #
    #
    #
    #
    # def main(self):
    #
    #     window_size = 60
    #     lstm_model = build_train_lstm_model(date, window_size)
    #
    #
    #
    #
    #
    #
    #     preprocess_data(df['Open'], 60)
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # #Inverse scaling
    # # train_predict = scaler_y.inverse_transform(train_predict)
    # # y_train = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    # # test_predict = scaler_y.inverse_transform(test_predict)
    # # y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # symbol = 'EURUSD'
    # lot_size = 10000
    # take_profit = predicted_price *1.05 #in pips
    # stop_loss = (take_profit - predicted_price) / 4 #in pips
    #
    # df['Dates'] = pd.to_datetime(df['Dates'])
    # #print(pd.unique(df['Dates'].dt.date))
    #
    # #loop through each day of market data
    # for day in pd.unique(df['Dates'].dt.date):
    #
    #     day_df = df[df['Dates'].dt.date == day].reset_index(drop=True)
    #
    #     if day_df.loc[0, 'Dates'].time() >= pd.Timestamp("09:00").time():
    #         start_index = 0
    #     else:
    #         start_index = day_df[day_df["Dates"].dt.time >= pd.Timestamp("09:00").time()].index[0]
    #
    #     # settle_index = start_index + 30
    #     #
    #     # high = np.max(day_df['High'][start_index:settle_index])
    #     # low = np.min(day_df['Low'][start_index:settle_index])
    #     #
    #     # #wait for the market price to rise above or below the high and low pice
    #     # for i in range(settle_index, len(day_df)):
    #     #     row = day_df.loc[i]
    #     #     if row['Open'] > high:
    #     #         entry_price = row["Open"]
    #     #         stop_loss_price = entry_price - stop_loss * 0.0001
    #     #         take_profit_price = entry_price + take_profit * 0.0001
    #     #         print("Buy signal executed at price: ", entry_price)
    #     #         break
    #     #
    #     #     elif row['Open'] < low:
    #     #         entry_price = row["Open"]
    #     #         stop_loss_price = entry_price + stop_loss * 0.0001
    #     #         take_profit_price = entry_price - take_profit * 0.0001
    #     #         print('Sell signal executed at price: ', entry_price)
    #     #         break
    #     #
    #     # #calculate pnl and return
    #     # if entry_price is not None:
    #     #     exit_price = day_df.iloc[i+1]["Open"]
    #     #     profit_loss = (exit_price - entry_price) * lot_size
    #     #     if profit_loss > 0:
    #     #         print("Profit:", profit_loss)
    #     #     else:
    #     #         print("Loss", profit_loss)
    #
    # end_time = pd.Timestamp('17:30').time()
    # end_index = day_df[day_df['Dates'].dt.time >= end_time].index[0] if len(day_df[day_df['Dates'].dt.time >= end_time]) >0 else len(day_df)
    #
    # day_df = day_df.iloc[start_index:end_index]
    #
    # if len(day_df) > 0:
    #     open_price = day_df.iloc[0]['Open']
    #     close_price = day_df.iloc[0]['Close']
    #     pct_change = (close_price - open_price) / open_price
    # else:
    #     pct_change = 0.0
    #
    # print('Percentage Change:', pct_change)
    #
    # plt.figure(figsize=(15,10))
    # plt.grid(True)
    # plt.title('Portfolio', fontsize=16, color='b')
    # plt.plot()
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # #wait 30mins for the market to settle
    #



