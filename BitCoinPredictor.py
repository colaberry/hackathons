import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/colaberry/data/master/Bitcoin/bitcoin_dataset.csv')
test = pd.read_csv('test_set.csv')
data.head()
data.shape
data.info()
data.describe()
num_missing = data.isnull().sum()
num_missing[num_missing >0]

data_new = data.fillna(method='ffill')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X = data_new.drop(['Date','btc_market_price'], axis =1)
y = data_new['btc_market_price']
test_set=test.drop(['Date'], axis =1)
from sklearn.model_selection import train_test_split
X_train_org, X_test_org, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 20)

X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)
#LINEAR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
lr = LinearRegression()
lr.fit(X_train, y_train)
ypred=lr.predict(X_test)
lr1 = LinearRegression()
lr1.fit(X,y)
y1pred=lr1.predict(test_set)
mse(y_test,ypred)
