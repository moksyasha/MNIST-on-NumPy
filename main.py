import gzip
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os
import requests


class CrossEntropy:
    def forward(self, y_true, y_pred):
        self.y_pred = y_pred
        self.y_true = np.zeros(10)
        self.y_true[y_true] = 1
        self.loss = -np.sum(self.y_true * np.log(y_pred))
        return self.loss

    def backward(self):
        dz = -self.y_true / self.y_pred
        return dz


class Softmax:
    def forward(self, x):
        x = x - np.max(x)
        self.p = np.exp(x) / np.sum(np.exp(x))
        return self.p

    def backward(self, dz):
        jacobian = np.diag(dz)  # for right dimension
        for i in range(len(jacobian)):
            for j in range(len(jacobian)):
                if i == j:
                    jacobian[i][j] = self.p[i] * (1 - self.p[j])
                else:
                    jacobian[i][j] = -self.p[i] * self.p[j]

        return np.matmul(dz, jacobian)  # shape 10 and 10-10


class ReLu:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dz):
        dz[self.x < 0] = 0
        return dz


class FC:
    def __init__(self, in_size, out_size):
        self.W = np.random.normal(
            scale=1, size=(out_size, in_size)) * np.sqrt(2 / (in_size + out_size))
        self.b = np.zeros(out_size)

    def forward(self, x):
        self.x = x
        if np.array(x).shape[0] != self.W.shape[1]:
            print('X is not the same dimention as in_size')
        return np.dot(self.W, self.x) + self.b

    def backward(self, dz, learning_rate=0.001):
        # dL/dFC * dFC/dW = dL/dFC * x
        self.dW = np.outer(dz, self.x)

        # dL/dFC
        self.db = dz

        # dL/dFC * w
        self.dx = np.dot(dz, self.W)

        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db

        return self.dx


class FirstNet:
    def __init__(self):
        self.f1_layer = FC(784, 128)
        self.a1_layer = ReLu()

        self.f2_layer = FC(128, 50)
        self.a2_layer = ReLu()

        self.f3_layer = FC(50, 10)
        self.a3_layer = Softmax()

    def forward(self, x):
        net = self.f1_layer.forward(x)
        net = self.a1_layer.forward(net)

        net = self.f2_layer.forward(net)
        net = self.a2_layer.forward(net)

        net = self.f3_layer.forward(net)    # shape 10
        net = self.a3_layer.forward(net)

        return net

    def backward(self, dz, learning_rate=0.01):
        # dz from CE ( dL / dSoftmax )
        dz = self.a3_layer.backward(dz)     # dz from Softmax ( dL / dFC )
        dz = self.f3_layer.backward(dz, learning_rate)  # dz from FC ( dL / dX )

        dz = self.a2_layer.backward(dz)
        dz = self.f2_layer.backward(dz, learning_rate)

        dz = self.a1_layer.backward(dz)
        dz = self.f1_layer.backward(dz, learning_rate)

        return dz

    def save(self, path):
        np.savez(path, w1=self.f1_layer.W, w2=self.f2_layer.W, w3=self.f3_layer.W,
                       b1=self.f1_layer.b, b2=self.f2_layer.b, b3=self.f3_layer.b)

    def load(self, path):
        with np.load(path) as data:
            self.f1_layer.W = data['w1']
            self.f1_layer.b = data['b1']
            self.f2_layer.W = data['w2']
            self.f2_layer.b = data['b2']
            self.f3_layer.W = data['w3']
            self.f3_layer.b = data['b3']


def compute_acc(x_test, y_test, net):
    acc = 0.0
    for i in range(len(x_test)):
        y_h = net.forward(x_test[i])
        y = np.argmax(y_h)
        if y == y_test[i]:
            acc += 1.0
    return acc / len(y_test)


def fit(epochs, model, learning_rate, loss_func, x_train, y_train, x_test, y_test):
    loss_tr_arr = []
    loss_val_arr = []
    acc_tr_arr = []
    acc_val_arr = []

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")

        loss_tr = 0
        loss_val = 0

        sh = list(range(len(x_train)))
        np.random.shuffle(sh)

        for i in range(len(x_train)):
            x = x_train[sh[i]]
            y = y_train[sh[i]]
            pred = model.forward(x)
            loss = loss_func.forward(y, pred)
            loss_tr += loss

            dz = loss_func.backward()
            dz = model.backward(dz, learning_rate)

        loss_tr /= len(x_train)
        loss_tr_arr.append(loss_tr)

        acc_train = compute_acc(x_train, y_train, model)
        acc_tr_arr.append(acc_train)

        for i in range(len(x_test)):
            x = x_test[i]
            y = y_test[i]
            pred = model.forward(x)
            loss = loss_func.forward(y, pred)
            loss_val += loss

        loss_val /= len(x_test)
        loss_val_arr.append(loss_val)

        acc_val = compute_acc(x_test, y_test, model)
        acc_val_arr.append(acc_val)

        print(f"Train Loss: {loss_tr} / Acc: {acc_train} ")
        print(f"Valid Loss: {loss_val} / Acc: {acc_val} ")
        #print(epoch, loss_tr, loss_val, acc_train, acc_val)

    show_graphs(loss_tr_arr, "Loss train", 1)
    show_graphs(loss_tr_arr, "Loss valid", 2)
    show_graphs(acc_tr_arr, "Acc train", 3)
    show_graphs(acc_val_arr, "Acc valid", 4)
    plt.show()


def fetch(url, path):
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


def show_graphs(array, name, num_fig):
    plt.figure(num_fig, figsize=(8, 8))
    plt.plot(array)
    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.title(name)
    # включаем основную сетку
    plt.grid(which='major')
    # включаем дополнительную сетку
    plt.grid(which='minor', linestyle=':')


def show_examples(model, x_test, y_test, is_show_graph):
    plt.figure(1, figsize=(8, 8))
    gt = []
    pred = []
    for ind in range(1, 26):
        random_number = int(np.random.uniform(0, x_test.shape[0]))
        plt.subplot(5, 5, ind)
        predicted = np.argmax(model.forward(x_test[random_number]))
        plt.title("Pred to be " + str(predicted))
        plt.imshow(x_test[random_number].reshape(28, 28), cmap=plt.get_cmap('gray'))
        plt.axis('off')
        gt.append(y_test[random_number])
        pred.append(predicted)

    if is_show_graph:
        plt.figure(2, figsize=(8, 6))
        plt.subplot(1, 1, 1)
        plt.axis('on')
        plt.plot(range(1, 26), gt, marker='D', mfc='green', mec='green', ms='10', lw=0.0)
        plt.plot(range(1, 26), pred, marker='o', mfc='red', mec='red', ms='7', lw=0.0)

        # включаем основную сетку
        plt.grid(which='major')
        plt.title("Ground Truth V/s Predicted")
        plt.legend(['gt', 'pred'])

    plt.show()


def main():
    path = './data'
    X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", path)[0x10:].reshape((-1, 28*28))
    Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", path)[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", path)[0x10:].reshape((-1, 28 * 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", path)[8:]

    X = X / 255
    X_test = X_test / 255

    print("X shape: ", X.shape)
    print("Y shape: ", Y.shape)
    print("Xtest shape: ", X_test.shape)
    print("Ytest shape: ", Y_test.shape)

    loss_func = CrossEntropy()
    model = FirstNet()
    learning_rate = 0.001
    epochs = 10
    show_examples(model, X_test, Y_test, 1)

    #fit(epochs, model, learning_rate, loss_func, X, Y, X_test, Y_test)
    #model.save('./weights/save.npz')
    model.load('./weights/save.npz')
    show_examples(model, X_test, Y_test, 1)


if __name__ == '__main__':
    main()
