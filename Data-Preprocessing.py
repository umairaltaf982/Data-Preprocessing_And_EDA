import os
import glob
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import iqr
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data_dir = "raw/"
electricity_path = os.path.join(data_dir, "electricity_raw_data/*.json")
weather_path = os.path.join(data_dir, "weather_raw_data/*.csv")

def load_json_files(file_paths):
    dataframes = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data['response']['data'])
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def load_csv_files(file_paths):
    return pd.concat([pd.read_csv(f) for f in file_paths], ignore_index=True)

electricity_files = glob.glob(electricity_path)
weather_files = glob.glob(weather_path)

electricity_data = load_json_files(electricity_files)
weather_data = load_csv_files(weather_files)

if 'date' in weather_data.columns:
    weather_data.rename(columns={'date': 'datetime'}, inplace=True)

electricity_data['period'] = pd.to_datetime(electricity_data['period'], format='%Y-%m-%dT%H')
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'], errors='coerce')
weather_data['datetime'] = weather_data['datetime'].dt.tz_localize(None)

data = pd.merge(electricity_data, weather_data, left_on='period', right_on='datetime', how='left')

def missing_data_report(df):
    return df.isnull().sum() / len(df) * 100

missing_report = missing_data_report(data)
print("Missing Data Report:\n", missing_report)

data.drop_duplicates(inplace=True)

data['value'] = pd.to_numeric(data['value'], errors='coerce')

data['hour'] = data['period'].dt.hour
data['day_of_week'] = data['period'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

Q1 = data['value'].quantile(0.25)
Q3 = data['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outlier_indices_iqr = data[(data['value'] < lower_bound) | (data['value'] > upper_bound)].index
data_clean = data.drop(outlier_indices_iqr)

result = seasonal_decompose(data_clean.set_index('period')['value'], model='additive', period=24)
result.plot()
plt.show()

adf_test = adfuller(data_clean['value'].dropna())
print("ADF Test Statistic:", adf_test[0])
print("p-value:", adf_test[1])

data_clean.to_csv(os.path.join(data_dir, "cleaned_data.csv"), index=False)

X = data_clean[['hour', 'day_of_week', 'is_weekend']]
y = data_clean['value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

plt.figure(figsize=(12, 6))
plt.plot(data_clean['period'], data_clean['value'], label='Electricity Demand')
plt.xlabel('Time')
plt.ylabel('Electricity Demand')
plt.title('Electricity Demand Over Time')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data_clean['value'], bins=30, kde=True)
plt.xlabel('Electricity Demand')
plt.ylabel('Frequency')
plt.title('Distribution of Electricity Demand')
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(data_clean.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

metrics_report = f"MSE: {mse}\nRMSE: {rmse}\nR2 Score: {r2}"
with open(os.path.join(data_dir, "model_metrics.txt"), 'w') as f:
    f.write(metrics_report)
