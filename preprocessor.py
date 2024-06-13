import pandas as pd 

# Bitcoin and Google datasets are loaded
coindata = pd.read_csv('Data/CoinMarketCapWebApp.csv')
googledata = pd.read_csv('Data/GoogleVolume.csv')

# Loaded raw datasets are printed
print(coindata.head())
print(googledata.head())

# Unused column are dropped and preprocessed
coindata = coindata.drop(['#'], axis=1)
coindata.columns = ['Date','Open','High','Low','Close','Volume']
googledata = googledata.drop(['Date','#'], axis=1)

# final results are printed for two datasets
print(coindata.head())
print(googledata.head())

# Finally two datasets are concatenated
last = pd.concat([coindata,googledata], axis=1)
print(last.head())

# Final dataset exported 
last.to_csv('Cleaned Data/Bitcoin3D.csv', index=False)