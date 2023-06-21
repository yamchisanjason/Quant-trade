import pandas as pd
import numpy as np
import quantstats as qs
import stats
import datetime
from datetime import datetime, date, time
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

class Hans123:
    def __init__(self, initcap=None):
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.initcap = initcap




    def preprocess_data(self, data, window_size):
        X, y = [], []
        for i in range(len(X) - window_size):
            v = X.iloc[i: (i + window_size)].values
            X.append(v)
            y.append(data[i+window_size])
        return np.array(X), np.array(y)



    def build_train_lstm_model(self, data, window_size, epochs=100):
        # create dataset
        X, y = preprocess_data(data, window_size)
        X = X.reshape(X.shape[0], X.shape[1])

        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df[['Open', 'Close', 'High', 'Low']])

        # train-test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[0:train_size], X[train_size:len(X)]
        y_train, y_test = y[0:train_size], y[train_size:len(y)]

        # LSTM model
        model = Sequential()
        model.add(LSTM(4, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=2)

        # Evaluate the model
        mse = model.evaluate(X_test, y_test, verbose=0)
        print('Mean Squared Error:', mse)

        # Predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Generate predictions for next day's market price
        last_30_days = df_scaled[-30:]
        last_30_days = last_30_days.reshape((1, last_30_days.shape[0], last_30_days.shape[1]))
        predicted_price = model.predict(last_30_days)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]


    def main(self):
        #read csv
        data = pd.read_csv(r'C:\Users\jason.yam\EURUSDmins_data.csv')
        df = data.dropna()
        df['Dates'] = pd.to_datetime(df['Dates'], format="%d/%m/%Y %H:%M").dt.strftime("%Y.%m.%d %H:%M")
        # print(df)

        window_size = 60
        lstm_model = build_train_lstm_model(date, window_size)






        preprocess_data(df['Open'], 60)









    #Inverse scaling
    # train_predict = scaler_y.inverse_transform(train_predict)
    # y_train = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    # test_predict = scaler_y.inverse_transform(test_predict)
    # y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))











    symbol = 'EURUSD'
    lot_size = 10000
    take_profit = predicted_price *1.05 #in pips
    stop_loss = (take_profit - predicted_price) / 4 #in pips

    df['Dates'] = pd.to_datetime(df['Dates'])
    #print(pd.unique(df['Dates'].dt.date))

    #loop through each day of market data
    for day in pd.unique(df['Dates'].dt.date):

        day_df = df[df['Dates'].dt.date == day].reset_index(drop=True)

        if day_df.loc[0, 'Dates'].time() >= pd.Timestamp("09:00").time():
            start_index = 0
        else:
            start_index = day_df[day_df["Dates"].dt.time >= pd.Timestamp("09:00").time()].index[0]

        # settle_index = start_index + 30
        #
        # high = np.max(day_df['High'][start_index:settle_index])
        # low = np.min(day_df['Low'][start_index:settle_index])
        #
        # #wait for the market price to rise above or below the high and low pice
        # for i in range(settle_index, len(day_df)):
        #     row = day_df.loc[i]
        #     if row['Open'] > high:
        #         entry_price = row["Open"]
        #         stop_loss_price = entry_price - stop_loss * 0.0001
        #         take_profit_price = entry_price + take_profit * 0.0001
        #         print("Buy signal executed at price: ", entry_price)
        #         break
        #
        #     elif row['Open'] < low:
        #         entry_price = row["Open"]
        #         stop_loss_price = entry_price + stop_loss * 0.0001
        #         take_profit_price = entry_price - take_profit * 0.0001
        #         print('Sell signal executed at price: ', entry_price)
        #         break
        #
        # #calculate pnl and return
        # if entry_price is not None:
        #     exit_price = day_df.iloc[i+1]["Open"]
        #     profit_loss = (exit_price - entry_price) * lot_size
        #     if profit_loss > 0:
        #         print("Profit:", profit_loss)
        #     else:
        #         print("Loss", profit_loss)

    end_time = pd.Timestamp('17:30').time()
    end_index = day_df[day_df['Dates'].dt.time >= end_time].index[0] if len(day_df[day_df['Dates'].dt.time >= end_time]) >0 else len(day_df)

    day_df = day_df.iloc[start_index:end_index]

    if len(day_df) > 0:
        open_price = day_df.iloc[0]['Open']
        close_price = day_df.iloc[0]['Close']
        pct_change = (close_price - open_price) / open_price
    else:
        pct_change = 0.0

    print('Percentage Change:', pct_change)

    plt.figure(figsize=(15,10))
    plt.grid(True)
    plt.title('Portfolio', fontsize=16, color='b')
    plt.plot()








    if __name__ == '__main__':
        main()



    #wait 30mins for the market to settle




