# Quantitative Financial Sector Analysis – Final Fix with Correct Factor Columns

# --- Imports ---
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import statsmodels.api as sm
from fredapi import Fred
import io
import zipfile
import requests

# --- Step 1: Download Selected Financial Stocks ---
stocks = ['JPM', 'BLK', 'SCHW']
data = yf.download(stocks, start='2015-01-01', end='2025-01-01')['Close']
data = data.dropna()

# --- Step 2: Calculate Daily Returns ---
returns = data.pct_change().dropna()

# --- Step 3: Add Interest Rate Proxy from FRED ---
fred = Fred(api_key='16ab3af2c8714526cc67b9b4dcfc77cb')
t10y2y = fred.get_series('T10Y2Y', start='2015-01-01', end='2025-01-01')
t10y2y = t10y2y.resample('B').ffill().reindex(returns.index).ffill()
returns['RateProxy'] = t10y2y / 100

# --- Step 4: Rolling Beta of JPM vs Rate Proxy ---
window = 252
betas = []
for i in range(window, len(returns)):
    y = returns['JPM'].iloc[i-window:i]
    X = returns['RateProxy'].iloc[i-window:i]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    betas.append(model.params.iloc[1])

# Plot Rolling Beta
plt.figure(figsize=(10,5))
plt.plot(returns.index[window:], betas)
plt.title('Rolling Beta of JPM vs T10Y2Y Spread')
plt.xlabel('Date')
plt.ylabel('Beta')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# --- Step 5: Fama-French Factor Regression (Final Fix) ---
ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
r = requests.get(ff_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
csv_name = z.namelist()[0]

lines = z.read(csv_name).decode("latin1").splitlines()
data_start = next(i for i, line in enumerate(lines) if line.strip().startswith("1926"))
header_line = lines[data_start - 1].strip().split()
data_end = next((i for i, line in enumerate(lines) if line.strip().startswith("99")), len(lines))

factors = pd.read_csv(
    io.StringIO("\n".join(lines[data_start:data_end])),
    header=None,
    names=["Date"] + header_line
)

factors['Date'] = pd.to_datetime(factors['Date'], format='%Y%m%d', errors='coerce')
factors = factors.dropna(subset=['Date'])
factors = factors.set_index('Date')
factors.index = pd.to_datetime(factors.index)
factors = factors.sort_index()

# Rename columns to match model inputs
factors.columns = [col.replace(' ', '').replace('.', '').replace('-', '') for col in factors.columns]
factors = factors.rename(columns={
    'MktRF': 'Mkt-RF',
    'SMB': 'SMB',
    'HML': 'HML',
    'RF': 'RF'
})

# Debug
print("Final factor columns:", list(factors.columns))
print("Index range:", factors.index.min(), "to", factors.index.max())

factors = factors.loc['2015-01-01':'2025-01-01']

# --- Step 6: Run Regressions ---
tickers = ['JPM', 'BLK', 'SCHW']
for ticker in tickers:
    ff_data = pd.merge(returns[[ticker]], factors[['Mkt-RF', 'SMB', 'HML', 'RF']], left_index=True, right_index=True)
    ff_data['Excess_Return'] = ff_data[ticker] - ff_data['RF']
    X = sm.add_constant(ff_data[['Mkt-RF', 'SMB', 'HML']])
    y = ff_data['Excess_Return']
    model = sm.OLS(y, X).fit()
    print(f"\nFama-French Regression for {ticker}:")
    print(model.summary())

# --- Step 7: PCA on Firm Fundamentals ---
fundamentals = pd.DataFrame({
    'ROE': [12, 15, 10],
    'Leverage': [10, 9, 13],
    'EPS_Growth': [5, 6, 3],
    'Volatility': [0.25, 0.22, 0.28]
}, index=['JPM', 'BLK', 'SCHW'])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(fundamentals)
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=fundamentals.index)

# --- Step 8: Clustering ---
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(pca_df)
pca_df['Cluster'] = kmeans.labels_

# Plot PCA Clusters
plt.figure(figsize=(8,5))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set2')
plt.title('PCA Clustering of JPM, BLK, SCHW')
plt.tight_layout()
plt.show(block=False)

# --- Step 9: Backtest Long/Short Strategy ---
cluster_0 = pca_df[pca_df['Cluster'] == 0].index
cluster_1 = pca_df[pca_df['Cluster'] == 1].index
long_returns = returns[cluster_0].mean(axis=1) if not cluster_0.empty else 0
short_returns = -returns[cluster_1].mean(axis=1) if not cluster_1.empty else 0
strategy_returns = (long_returns + short_returns) / 2

# Cumulative Performance
cumulative = (1 + strategy_returns).cumprod()
plt.figure(figsize=(10,5))
cumulative.plot()
plt.title('Backtested Long/Short Strategy on JPM, BLK, SCHW')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# Keep plots visible until user closes them manually
input("\nPress ENTER to exit after reviewing all plots...")
