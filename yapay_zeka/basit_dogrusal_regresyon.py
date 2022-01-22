import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("veri_setleri/dogrusal_regresyon_veriseti.csv",sep=";")
print(data.head(5))

plt.scatter(x=data.deneyim,y=data.maas)
plt.xlabel("deneyim")
plt.ylabel("maaş")
plt.show()

lr=LinearRegression()
X=data.deneyim.values.reshape(-1,1)
y=data.maas.values.reshape(-1,1)
lr.fit(X,y)

a=lr.coef_
b=lr.intercept_
print(f"a={a},b={b}")

tahmin = lr.predict([[9],[11],[16]])
print(tahmin)


