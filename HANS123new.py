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






    # def preprocess_data(self, data, window_size):
    #     X, y = [], []
    #     for i in range(len(X) - window_size):
    #         v = data[i: (i + window_size)].values
    #         X.append(v)
    #         y.append(data[i+window_size])
    #     return np.array(X), np.array(y)
    #     print('hello1')
    #
    #
    # def build_train_lstm_model(self, data, window_size, epochs=100):
    #     # # create dataset
    #     # X, y = preprocess_data(data, window_size)
    #     # X = X.reshape(X.shape[0], X.shape[1])
    #
    #
    #     #preprocess data
    #     scaler = MinMaxScaler()
    #     data_scaled = scaler.fit_transform(data[['Open', 'Close', 'High', 'Low']])
    #     print('hello2')
    #
    #     #create dataset
    #     X, y = self.preprocess_data(data_scaled, window_size)
    #     print('hello3')
    #
    #     # train-test split
    #     train_size = int(len(X) * 0.8)
    #     X_train, X_test = X[0:train_size], X[train_size:len(X)]
    #     y_train, y_test = y[0:train_size], y[train_size:len(y)]
    #     print('hello4')
    #
    #     # LSTM model
    #     model = Sequential()
    #     model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    #     model.add(Dense(1))
    #     model.add(Dropout(0.2))
    #     model.compile(loss='mean_squared_error', optimizer='adam')
    #     model.fit(X_train, y_train, epochs=epochs, batch_size=32)
    #     print('hello5')
    #
    #     # Generate predictions for next day's market price
    #     last_30_days = data_scaled[-30:]
    #     last_30_days = last_30_days.reshape((1, last_30_days.shape[0], last_30_days.shape[1]))
    #     predicted_price = model.predict(last_30_days)
    #     predicted_price = scaler.inverse_transform(predicted_price)[0][0]
    #     print('hello6')
    #     return predicted_price

    def trading_strategy(self, data):
        # set initial values
        capital = self.initcap
        take_profit = 0
        stop_loss = 0
        position_size = capital * 0.1
        start_time = None
        end_time = None
        high = 0
        low = np.inf
        entry_price = None
        capital_values = []
        dates = []
        num_trades = 0
        num_winningtrades = 0

        # loop-through all the data
        i = 0
        while i < len(data):
            # wait for 30 mins and record high and low prices
            current_time = datetime.strptime(data['Dates'].iloc[i], "%Y.%m.%d %H:%M")
            #print('9')
            if current_time.time() >= time(hour=9) and current_time.time() <= time(hour=9, minute=30):
                start_time = current_time
                end_time = start_time + timedelta(minutes=30)
                high = 0
                low = np.inf
                print(start_time)
                print(end_time)


                while current_time < end_time and i < len(data):
                    current_time = datetime.strptime(data['Dates'].iloc[i], "%Y.%m.%d %H:%M")


                    high = max(high, data['High'].iloc[i])
                    low = min(low, data['Low'].iloc[i])

                    i += 1
                print(high)
                print(low)
                print('-')

                if data['Close'].iloc[i] >= high:
                    entry_price = high
                    stop_loss = low
                    take_profit = entry_price + (self.rr_ratio) * (entry_price - stop_loss)
                    profit = position_size * (take_profit - entry_price)
                    capital += profit
                    print('Buy signal generated at', data['Dates'].iloc[i ], 'with entry price', entry_price, '\ncumulated capital:', capital)

                elif data['Close'].iloc[i ] <= low:
                    entry_price = low
                    stop_loss = high
                    take_profit = entry_price - (self.rr_ratio * (stop_loss - entry_price))
                    loss = position_size * (stop_loss - entry_price)
                    capital -= loss
                    print('Sell signal generated at', data['Dates'].iloc[i ], 'with entry price', entry_price, '\ncumulated capital:', capital)

                if entry_price is not None:
                    #check if take profit or stop loss is hit
                    while i < len(data):
                        current_time = datetime.strptime(data['Dates'].iloc[i], "%Y.%m.%d %H:%M")
                        if current_time.time() >= time(hour=17, minute=30):
                            break

                        if data['High'].iloc[i] >= take_profit:
                            exit_price = take_profit
                            print('Take profit hit at', data['Dates'].iloc[i], 'with exit price', exit_price)

                            break

                        elif data['Low'].iloc[i] <= stop_loss:
                            exit_price = stop_loss
                            print('Stop loss hit at', data['Dates'].iloc[i], 'with exit price', exit_price)

                            break



                        i += 1

                    #calculate return on trade and update capital
                    if exit_price is not None:
                        if entry_price < exit_price:
                            return_pct = (exit_price - entry_price) / entry_price
                            print('Trade return:', return_pct, 'with entry capital:', capital)
                            capital += capital * return_pct
                            num_winningtrades += 1

                        else:
                            return_pct = (entry_price - exit_price) / entry_price
                            print('Trade return:', -return_pct, 'with entry capital:', capital)
                            capital += capital * return_pct
                            num_winningtrades += 1

                        capital_values.append(capital)
                        dates.append(data['Dates'].iloc[i-1])
                        num_trades += 1

                entry_price = None

            i += 1
        print(num_trades)
        print(num_winningtrades)
        if num_trades > 0:
            win_rate = num_winningtrades / num_trades
            print('Win rate:', win_rate)
        else:
            print('No trades executed.')



        print(capital_values)
        #plot the cumulated capital over time
        plt.plot(dates, capital_values)
        plt.xlabel('Time')
        plt.ylabel('Cumulated Capital')
        plt.xticks(rotation=45)
        plt.show()




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
        sharpe_ratio = (returns - 0.0377) / std        #assume risk-free rate = 3.77%

        print('Final Capital:', capital)
        print('Entry Price:', self.initcap)
        print('Returns:', returns)
        print('Daily Returns:', daily_returns)
        print('Mean:', mean)
        print('Standard Deviation:', std)
        print('Sharpe Ratio:', sharpe_ratio)

        return capital, entry_price, take_profit, stop_loss


if __name__ == '__main__':
    strategy = Hans123(initcap=10000)
    capital, entry_price, take_profit, stop_loss = strategy.run_strategy(r'C:\Users\jason.yam\EURUSDmins_data.csv')









  
