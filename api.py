import http.server
import socketserver
import json
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def learn_location(location, access_points):
    files = [os.path.join("locations", file)
             for file in os.listdir("locations")]
    if f"locations/{location}.txt" in files:
        with open(f"locations/{location}.txt", "a") as f:
            f.write(json.dumps(access_points) + "\n")
    else:
        with open(f"locations/{location}.txt", "w") as f:
            f.write(json.dumps(access_points) + "\n")


def predict_location(access_points):
    files = [os.path.join("locations", file)
             for file in os.listdir("locations")]
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

    test_json = access_points

    X_test = []
    for key in all_keys:
        signal = test_json.get(key)
        X_test.append(signal if signal else 0)

    results = []

    gnb = GaussianNB()
    y_pred = gnb.fit(X, y)
    results.append(y_pred.predict([X_test])[0])

    X, y = np.array(X), np.array(y)
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(X, y)
    results.append(neigh.predict([X_test])[0])

    dt_classifier = DecisionTreeClassifier(random_state=4)
    dt_classifier.fit(X, y)
    results.append(dt_classifier.predict([X_test])[0])

    return files[results.index(max(set(results), key=results.count))]


class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            response = {'message': 'Hello, World!'}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            post_data = json.loads(post_data)
            print(f"Received POST data: {post_data}")

            if not "access_points" in post_data:
                response = {'message': 'Access points not specified'}
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return

            access_points = post_data["access_points"]

            print(f"Detecting location: {access_points}")

            response = {'message': 'Data received'}
            if access_points:
                print("Predicting location")
                location = predict_location(access_points)
                response['location'] = location

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            return

        if self.path == '/api/learn':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            print(f"Received POST data: {post_data}")

            print(type(post_data))
            post_data = json.loads(str(post_data))

            if not "location" in post_data:
                response = {'message': 'Location not specified'}
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return
            if not "access_points" in post_data:
                response = {'message': 'Access points not specified'}
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return

            location = post_data["location"]
            access_points = post_data["access_points"]

            print(f"Learning location: {location}")

            if access_points:
                learn_location(location, access_points)

            print(f"Learning complete")

            response = {'message': 'Data received and learning'}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')


PORT = 8012
HOST = "0.0.0.0"

with socketserver.TCPServer((HOST, PORT), MyHandler) as httpd:
    print("Serving at port", PORT)
    httpd.serve_forever()
