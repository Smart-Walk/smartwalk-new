import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

files = [
    "300.txt",
    "313.txt",
]
# files = ["108.txt", "654.txt", "1633.txt"]
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
test_json = {
    "NEXTDOOR Megatron_3 58:D5:6E:EB:33:0B": 90,
    "NEXTDOOR Megatron_3 30:4F:75:4A:EF:B1": 85,
    "NEXTDOOR Megatron_3 AC:15:A2:B9:70:87": 79,
    " AE:15:A2:A9:70:87": 72,
    "309 - The NextDoor_2.4Ghz 34:60:F9:FC:25:98": 70,
    "OPPO A9 2020 BE:C7:23:F5:68:A1": 70,
    "NEXTDOOR Megatron_3 AC:15:A2:B9:70:86": 70,
    " AE:15:A2:26:8B:05": 67,
    "Vaishnavi 5th floor  E8:48:B8:59:C6:17": 62,
    " B6:B0:24:ED:57:88": 59,
    " 1E:61:B4:A6:A2:7F": 57,
    "The Next Door 3rd floor 1C:61:B4:86:A2:7F": 57,
    "NEXTDOOR Megatron_2 58:D5:6E:EB:33:6B": 57,
    "NEXTDOOR Megatron_4 10:27:F5:AF:F3:88": 55,
    "NEXTDOOR Megatron_3 58:D5:6E:EB:33:09": 55,
    " 1E:61:B4:A6:9E:F2": 54,
    "NEXTDOOR Megatron_4 30:4F:75:4B:0A:E0": 52,
    "503- NEXTDOOR Megatron A8:42:A1:89:45:54": 52,
    "Vaishnavi 3rd floor  B0:A7:B9:36:7F:B5": 52,
    "Anugraha 5th floor  34:60:F9:9C:24:74": 52,
    "305- The NextDoor 1C:61:B4:86:9D:8A": 52,
    "NEXTDOOR Megatron_2 B4:B0:24:DD:58:15": 52,
    " B6:B0:24:ED:58:15": 52,
    " 1E:61:B4:A6:A1:E6": 52,
    "NEXTDOOR Megatron_2 18:0F:76:FF:66:78": 50,
    "NEXTDOOR Megatron_2 58:D5:6E:EB:32:ED": 50,
    "Anugraha pg new 2nd floor 58:D5:6E:EB:32:FF": 50,
    "NEXTDOOR Megatron_1 0C:B6:D2:22:F1:24": 50,
    " 52:91:E3:25:03:92": 50,
    " AA:42:A1:A9:45:6F": 50,
    " 1A:EB:B6:88:11:5E": 50,
    "NEXTDOOR Megatron_1 34:60:F9:85:21:4B": 50,
    " B4:B0:24:DD:57:88": 50,
    " 9E:A2:F4:AF:35:14": 50,
    "NEXTDOOR Megatron_1 18:0F:76:80:FD:DE": 50,
    "309 - The NextDoor 34:60:F9:FC:25:9A": 50,
    "NEXTDOOR Megatron_3 30:4F:75:4A:EF:B0": 49,
    "Vaishnavi 4th floor  B0:A7:B9:36:81:38": 49,
    "204 - The NextDoor 1C:61:B4:86:96:C1": 49,
    "Anugraha pg new 4th floor 34:60:F9:FC:31:2C": 49,
    "Anugraha pg 2nd floor 58:D5:6E:ED:E2:C2": 49,
    "CHETHANA A  94:E3:EE:05:F5:3E": 49,
    "504-NEXTDOOR Megatron A8:42:A1:89:45:6F": 49,
    "Anugraha 3rd flore 14:EB:B6:88:11:5E": 49,
    " AA:42:A1:A9:45:4E": 49,
    " 9E:A2:F4:AF:3B:44": 49,
    "The Next Door 2nd floor 1C:61:B4:86:A7:CB": 49,
    "308 - The NextDoor 9C:A2:F4:8F:50:DD": 49,
    "The NEXTDOOR 50:91:E3:55:A9:C1": 49,
    "Anugraha PG 3rdfloor 58:D5:6E:BC:B6:DE": 49,
    " 58:D5:6E:EE:1B:0E": 49,
    "Anugraha PG 1st floor BC:0F:9A:6B:67:AB": 49,
    "Anugraha 4th floor 80:37:73:A2:35:0C": 49,
    "308 - The NextDoor 9C:A2:F4:8F:50:DC": 49,
    " 9E:A2:F4:AF:50:DC": 49,
    "307 - The NextDoor AC:15:A2:36:8B:07": 49,
    " AE:15:A2:56:8B:05": 49,
    " AE:15:A2:A9:70:86": 47,
    "NEXTDOOR Megatron_4 30:4F:75:4B:0A:E1": 47,
    "Anugraha pg new 3rd floor_5G 10:27:F5:7D:6C:1D": 45,
    "NEXTDOOR Megatron_4 10:27:F5:AF:F3:87": 44,
    " 1E:61:B4:A6:A1:E5": 42,
    "208 - The NextDoor 1C:61:B4:86:9E:F1": 40,
    " 1E:61:B4:A6:9E:F1": 40,
    "Anugraha pg new 4th floor-5G 34:60:F9:FC:31:2E": 39,
    "NEXTDOOR Megatron_2 5G 18:0F:76:FF:66:76": 39,
    "Vaishnavi 4th floor_5G B0:A7:B9:36:81:3A": 39,
    " B4:B0:24:DD:57:87": 39,
    " B6:B0:24:ED:57:87": 39,
    "NEXTDOOR Megatron_1 30:4F:75:4B:1D:91": 39,
    "NEXTDOOR Megatron_2 B4:B0:24:DD:58:14": 37,
    " B6:B0:24:ED:58:14": 37,
}
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


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


class ShallowNN(nn.Module):
    def __init__(self, input_size):
        super(ShallowNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


model = ShallowNN(X_train.shape[1])


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.view(-1, 1))
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor.view(-1, 1))
        val_preds = (val_outputs > 0.5).float()
        val_accuracy = accuracy_score(y_val_tensor.numpy(), val_preds.numpy())

    print(
        f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}"
    )


X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = (test_outputs > 0.5).float()

print(test_preds.numpy())
