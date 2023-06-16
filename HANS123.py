import pandas as pd
import numpy as np
import quantstats as qs
import stats
import datetime
from datetime import datetime, date, time
import time


df = pd.read_csv(r'C:\Users\jason.yam\EURUSDsec_data.csv')

df['Dates'] = pd.to_datetime(df['Dates'], format="%d/%m/%Y %H:%M").dt.strftime("%Y.%m.%d %H:%M")
#print(df)

symbol = 'EURUSD'
lot_size = 10000
stop_loss = 50 #in pips
take_profit = 100 #in pips

df['Dates'] = pd.to_datetime(df['Dates'])
#print(pd.unique(df['Dates'].dt.date))

#loop through each day of market data
for day in pd.unique(df['Dates'].dt.date):

    day_df = df[df['Dates'].dt.date == day].reset_index(drop=True)

    if day_df.loc[0, 'Dates'].time() >= pd.Timestamp("09:00").time():
        start_index = 0
    else:
        start_index = day_df[day_df["Dates"].dt.time >= pd.Timestamp("09:00").time()].index[0]

    settle_index = start_index + 30

    high = np.max(day_df['High'][start_index:settle_index])
    low = np.min(day_df['Low'][start_index:settle_index])

    #wait for the market price to rise above or below the high and low pice
    for i in range(settle_index, len(day_df)):
        row = day_df.loc[i]
        if row['Open'] > high:
            entry_price = row["Open"]
            stop_loss_price = entry_price - stop_loss * 0.0001
            take_profit_price = entry_price + take_profit * 0.0001
            print("Buy signal executed at price: ", entry_price)
            break

        elif row['Open'] < low:
            entry_price = row["Open"]
            stop_loss_price = entry_price + stop_loss * 0.0001
            take_profit_price = entry_price - take_profit * 0.0001
            print('Sell signal executed at price: ', entry_price)
            break

    #calculate pnl and return
    if entry_price is not None:
        exit_price = day_df.iloc[i+1]["Open"]
        profit_loss = (exit_price - entry_price) * lot_size
        if profit_loss > 0:
            print("Profit:", profit_loss)
        else:
            print("Loss", profit_loss)












#wait 30mins for the market to settle




