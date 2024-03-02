import numpy as np
import pandas as pd
import tables as tb
import zipfile
import os
import glob
import h5py
import tensorflow as tf
from tqdm import tqdm

class NASDAQ_Data():
    def __init__(self, time_range = ['2010-1-1', '2020-1-1']):
        self.etf_files = glob.glob('data/etfs/*.csv')
        self.stock_files = glob.glob('data/stocks/*.csv')
        self.time_range = time_range

    def get_meta(self):
        return pd.read_csv('data/symbols_valid_meta.csv')
    
    def get_ticker(self, ticker):

        def hawkes_process(data: pd.Series, kappa: float):
            assert(kappa > 0.0)
            alpha = np.exp(-kappa)
            arr = data.to_numpy()
            output = np.zeros(len(data))
            output[:] = np.nan
            for i in range(1, len(data)):
                if np.isnan(output[i - 1]):
                    output[i] = arr[i]
                else:
                    output[i] = output[i - 1] * alpha + arr[i]
            return np.array(output) * kappa
        
        def atr(high, low, close, n=14):
            tr = np.amax(np.vstack(((high - low).to_numpy(), (abs(high - close)).to_numpy(), (abs(low - close)).to_numpy())).T, axis=1)
            return pd.Series(tr).rolling(n).mean().to_numpy()
        
        def MinMaxNorm(X):
            '''
            Normalize data on scale (0, 1)
            '''
            return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

        if f"data/etfs/{ticker}.csv" in self.etf_files:
            df = pd.read_csv(f"data/etfs/{ticker}.csv").drop(columns = ['Adj Close'])
            df['Date'] = pd.to_datetime(df['Date'], format = "%Y-%m-%d")
            df = df.set_index('Date')
        elif f"data/stocks/{ticker}.csv" in self.stock_files:
            df = pd.read_csv(f"data/stocks/{ticker}.csv").drop(columns = ['Adj Close'])
            df['Date'] = pd.to_datetime(df['Date'], format = "%Y-%m-%d")
            df = df.set_index('Date')
        else:
            # print(f'{ticker} not found')
            return pd.DataFrame()
        
        # Introduce normalized features
        df.insert(len(df.columns), "ATR14", atr(df.High, df.Low, df.Close, n = 7), True)
        df.insert(len(df.columns), "Normalized", (MinMaxNorm(np.array(df['High'])) - MinMaxNorm(np.array(df['Low']))) / df['ATR14'], True)
        #df.insert(len(df.columns), "Hawkes0.1", hawkes_process(df['Normalized'], kappa = 0.1), True)
        df.insert(len(df.columns), "MINMAX", MinMaxNorm(np.array(df.Close)), True)
        df.insert(len(df.columns), "PCT_CHANGE", df.Close.pct_change(), True)

        if ( # check if 30 days before the minimum time_range exists
            len(df[pd.to_datetime(self.time_range[0], format = "%Y-%m-%d") - pd.Timedelta(30, "d") : pd.to_datetime(self.time_range[0], format = "%Y-%m-%d")]) != 0
            ):
            return df[
                pd.to_datetime(self.time_range[0], format = "%Y-%m-%d") : pd.to_datetime(self.time_range[1], format = "%Y-%m-%d")
            ]
        else:
            # print(f"{ticker} : Doesn't exist before {self.time_range[0]}")
            return pd.DataFrame()

    def create_dataset(self, filename):
        np.seterr(divide = 'ignore') 

        threshold = 0.20
        
        def rolling_window(arr, w_size, spacing):
            return [arr[i: i+w_size] for i in range(0, arr.shape[0] - w_size, spacing)]
        try:
            if not os.path.isfile(filename):
                fileh = tb.open_file(filename, mode = 'w')
            else:
                os.remove(filename)
                fileh = tb.open_file(filename, mode = 'w')

            hdf5_Features = fileh.create_earray(
                fileh.root,
                'Features',
                tb.Float64Atom(shape=()),
                (0, 200, 3),
                title = "Features",
                )
            
            hdf5_Labels = fileh.create_earray(
                fileh.root,
                'Labels',
                tb.Int8Atom(shape=()),
                (0, 3),
                title = "Labels",
                )
            
            symbols = self.get_meta().Symbol
            totals = [0, 0, 0]
            processed = 0
            invalid_symbols = []
            for symbol in tqdm(symbols):
                processed += 1
                # print(f'{processed}/{len(symbols)} : {symbol}')
                
                counts = [0, 0, 0]
                features = [] # shape = (# samples, 200 days, n features)
                target = [] # shape  = (# samples, label) label = [1, 0, 0]; [0, 1, 0]; [0, 0, 1]

                ticker = self.get_ticker(symbol)
                if not ticker.empty:
                    data = rolling_window(ticker.to_numpy(), 201, 10) # shape = (# samples, 210 days, 5 features)
                else:
                    invalid_symbols.append(symbol)
                    continue
                
                for sample in data:
                    if not np.isnan(sample).any():
                        if not np.isinf(sample).any():
                            future_10 = sample[-1:]
                            prior_200 = sample[:200]
                            last_day = prior_200[-1][3]
                            if np.average(future_10[:, 3]) >= last_day + (threshold * last_day): # did it go up by a threshold?
                                target.append([0, 0, 1])
                                counts[2] += 1
                            elif np.average(future_10[:, 3]) <= last_day - (threshold * last_day): # did it go down by a threshold?
                                target.append([1, 0, 0])
                                counts[0] += 1
                            else:
                                target.append([0, 1, 0])
                                counts[1] += 1

                            features.append(prior_200[:, [6, 7, 8]]) # keep only the TA indicators 


                # Balance Data
                balanced_features = []
                balanced_labels = []
                max_class = min(counts)
                if max_class == 0:
                    # Scrap the entry
                    invalid_symbols.append(symbol)
                    # print(f"{symbol} : Empty Class Present")
                    continue

                for i in range(3):
                    count = 0
                    for data, lab in zip(features, target):
                        if lab.index(1) == i:
                            balanced_features.append(data)
                            balanced_labels.append(lab)
                            count += 1
                        if count == max_class:
                            break
                    totals[i] += count

                hdf5_Features.append(np.array(balanced_features))
                hdf5_Labels.append(np.array(balanced_labels))

            fileh.close()
            print(f'Class Totals : {totals}')
            print("Didn't Pricess - ")
            print(invalid_symbols)
            np.seterr(divide = 'warn') 
        except Exception as e:
            print(e)
            try:
                fileh.close()
                os.remove(filename)
            except Exception as e:
                print(e)

