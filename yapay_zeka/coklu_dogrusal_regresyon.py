from numpy.lib.function_base import kaiser
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
veri = pd.read_csv("veri_setleri/coklu_dogrusal_regresyon_veriseti.csv",sep=";")
print(veri.head(3))
X = veri.iloc[:,[0,2]].values
y= veri.iloc[:,[1]].values.reshape(-1,1)
# print(y)
cdr = LinearRegression()
cdr.fit(X,y)
print(f"coef değerleri{cdr.coef_}")
print(f"intercept değeri{cdr.intercept_}")

kisiler=[[7,13],[19,39],[13,35]]
print(20*"*")
maaslar=cdr.predict(kisiler)
print(maaslar)
