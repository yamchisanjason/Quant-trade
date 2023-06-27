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
                print('8')

                while current_time < end_time and i < len(data):
                    current_time = datetime.strptime(data['Dates'].iloc[i], "%Y.%m.%d %H:%M")


                    high = max(high, data['High'].iloc[i])
                    low = min(low, data['Low'].iloc[i])

                    i += 1
                print(high)
                print(low)

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

                        else:
                            return_pct = (entry_price - exit_price) / entry_price
                            print('Trade return:', -return_pct, 'with entry capital:', capital)


                        capital_values.append(capital)
                        dates.append(data['Dates'].iloc[i-1])

                entry_price = None

            i += 1

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



