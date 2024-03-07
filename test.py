import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import os

files = [os.path.join("locations", file) for file in os.listdir("locations")]

datas = []
for file in files:
    with open(file, "r+") as f:
        datas.append(f.readlines())

for i in range(len(datas)):
    temp = [json.loads(j) for j in datas[i]]
    datas[i] = temp

all_keys = []

for inner_list in datas:
    for json_obj in inner_list:
        keys = json_obj.keys()
        for key in keys:
            if key.split()[0] in [
                "NEXTDOOR",
                "Infinix",
                "TP-Link_E1A8",
                "Vaishnavi",
                "Anugraha",
            ]:
                all_keys.append(key)


all_keys = sorted(list(set(all_keys)))
print(all_keys)
X = []
y = []
for cls, data in enumerate(datas):
    for jsons in data:
        data = []
        for key in all_keys:
            signal = jsons.get(key)
            data.append(signal if signal else 0)
        y.append(cls)
        X.append(data)
# print(all_keys)
# print(X)
test_json = {"Anugraha pg 2nd floor 58:d5:6e:ed:e2:c2": -72, "iPhone 12 02:75:06:33:a9:bb": -87, "NEXTDOOR Megatron_3 58:d5:6e:eb:33:09": -55, "NEXTDOOR Megatron_2 58:d5:6e:eb:33:69": -88, "Snehitha 14:eb:b6:c2:b8:35": -68, "NEXTDOOR Megatron_4 30:4f:75:4b:0a:e0": -72, "NEXTDOOR Megatron_3 30:4f:75:4a:ef:b0": -47, "NEXTDOOR Megatron_3 30:4f:75:4a:ef:b1": -77, "NEXTDOOR Megatron_2 5G 18:0f:76:ff:66:76": -78, "NEXTDOOR Megatron_2 18:0f:76:ff:66:78": -
             62, "Ashis OnePlus 62:61:d4:97:61:92": -71, "Anugraha pg 4thfloor-5ghz 58:d5:6e:eb:3f:db": -72, " aa:42:a1:a9:45:4e": -73, "NEXTDOOR Megatron_4 10:27:f5:af:f3:88": -79, "NEXTDOOR Megatron_3 ac:15:a2:b9:70:87": -65, "NEXTDOOR Megatron_3 ac:15:a2:b9:70:86": -73, "NEXTDOOR Megatron_2 58:d5:6e:eb:32:eb": -85, "NEXTDOOR Megatron_2 58:d5:6e:eb:32:ed": -57, "NEXTDOOR Megatron_3 58:d5:6e:eb:33:0b": -33, "The Next Door 3rd floor 1c:61:b4:86:a2:7f": -82}

X_test = []
for key in all_keys:
    signal = test_json.get(key)
    X_test.append(signal if signal else 0)

gnb = GaussianNB()
y_pred = gnb.fit(X, y)
print(y_pred.predict([X_test]))


X, y = np.array(X), np.array(y)
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X, y)
print(neigh.predict([X_test]))

dt_classifier = DecisionTreeClassifier(random_state=4)
dt_classifier.fit(X, y)
print(dt_classifier.predict([X_test]))

print(X_test in X, (X == X_test).all(1), y[-2])
