import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

import missingno as msno

import gc
import datetime
color = sns.color_palette()

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 100)

df = pd.read_csv('data.csv', encoding='ISO-8859-1') #latin-1 Encoding

df.rename(inplace=True, index=str, columns={
    'InvoiceNo': 'invoice_num',
    'StockCode' : 'stock_code',
    'Description' : 'description',
    'Quantity' : 'quantity',
    'InvoiceDate' : 'invoice_date',
    'UnitPrice' : 'unit_price',
    'CustomerID' : 'cust_id',
    'Country' : 'country'
})

df['invoice_date'] = pd.to_datetime(df.invoice_date, format='%m/%d/%Y %H:%M')
df['desciption'] = df.description.str.lower()
df_new = df.dropna()
df_new['cust_id'] = df_new['cust_id'].astype('int64')
df_new = df_new[df_new.quantity > 0]
df_new['amount_spent'] = df_new['quantity'] * df_new['unit_price']

#rearange columns
df_new = df_new[['invoice_num','invoice_date','stock_code','description','quantity','unit_price','amount_spent','cust_id','country']]

df_new.insert(loc=2, column='year_month', value=df_new['invoice_date'].map(lambda x: 100*x.year + x.month))
df_new.insert(loc=3, column='month', value=df_new.invoice_date.dt.month)