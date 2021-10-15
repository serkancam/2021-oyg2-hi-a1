import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

veri = pd.read_csv("veri_setleri/polinomsal_regresyon_veriseti.csv",sep=";")
print(veri.head(3))
X=veri.iloc[:,[0]].values
y=veri.iloc[:,[1]].values
lr=LinearRegression()
lr.fit(X,y)
y_tahmin=lr.predict(X)
# polinomal regresyon
pol_reg=PolynomialFeatures(degree=4)#4. dereceden polinom için ön işlem hazırla
X_pol=pol_reg.fit_transform(X) # hazırlanan ön işlemi X değerlerine uygula
lr_pol=LinearRegression()
lr_pol.fit(X_pol,y)
y_tahmin_pol=lr_pol.predict(X_pol)
# print(X_pol)
# plt.scatter(X,y)
# plt.plot(X,y_tahmin,color="red")
# plt.plot(X,y_tahmin_pol,color="green")
# plt.show()

#polinomal regresyon için tahmin nasıl yapılır
x_yeni=[[600],[700]]
x_yeni_pol=pol_reg.fit_transform(x_yeni)
hizlar=lr_pol.predict(x_yeni_pol)
print(hizlar)
