import cupy as cp
import time

# read MNIST label header from file
def read_label_headers(file_name):
    with open(file_name, 'rb') as f:
        magic = cp.frombuffer(f.read(4), dtype=cp.dtype('>i4'))
        num_labels = cp.frombuffer(f.read(4), dtype=cp.dtype('>i4'))
        return magic, num_labels

# read MNIST label data from file
def read_label_data(file_name):
    with open(file_name, 'rb') as f:
        f.read(8)
        labels = cp.frombuffer(f.read(), dtype=cp.dtype('B'))
        return labels

# read MNIST image header from file
def read_image_headers(file_name):
    with open(file_name, 'rb') as f:
        magic = cp.frombuffer(f.read(4), dtype=cp.dtype('>i4'))
        num_images = cp.frombuffer(f.read(4), dtype=cp.dtype('>i4'))
        num_rows = cp.frombuffer(f.read(4), dtype=cp.dtype('>i4'))
        num_cols = cp.frombuffer(f.read(4), dtype=cp.dtype('>i4'))
        return magic, num_images, num_rows, num_cols

# read MNIST image data from file
def read_image_data(file_name):
    with open(file_name, 'rb') as f:
        f.read(16)
        images = cp.frombuffer(f.read(), dtype=cp.dtype('B'))
        images = images.reshape(-1, 28*28)
        return images

def ReLU(Z):
    return cp.maximum(0, Z)

def softmax(Z):
    exps = cp.exp(Z - cp.max(Z))
    return exps / cp.sum(exps, axis=1, keepdims=True)

# forward propagation
def forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6):
    Z1 = cp.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = cp.dot(W2, A1) + b2
    A2 = ReLU(Z2)
    Z3 = cp.dot(W3, A2) + b3
    A3 = ReLU(Z3)
    Z4 = cp.dot(W4, A3) + b4
    A4 = ReLU(Z4)
    Z5 = cp.dot(W5, A4) + b5
    A5 = ReLU(Z5)
    Z6 = cp.dot(W6, A5) + b6
    A6 = softmax(Z6)

    return Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5, Z6, A6

# backward propagation
def backward_propagation(X, Y, Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5, Z6, A6, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6):
    m = X.shape[1]
    dZ6 = A6 - Y
    dW6 = cp.dot(dZ6, A5.T) / m
    db6 = cp.sum(dZ6, axis=1, keepdims=True) / m
    dZ5 = cp.dot(W6.T, dZ6) * (A5 > 0)
    dW5 = cp.dot(dZ5, A4.T) / m
    db5 = cp.sum(dZ5, axis=1, keepdims=True) / m
    dZ4 = cp.dot(W5.T, dZ5) * (Z4 > 0)
    dW4 = cp.dot(dZ4, A3.T) / m
    db4 = cp.sum(dZ4, axis=1, keepdims=True) / m
    dZ3 = cp.dot(W4.T, dZ4) * (Z3 > 0)
    dW3 = cp.dot(dZ3, A2.T) / m
    db3 = cp.sum(dZ3, axis=1, keepdims=True) / m
    dZ2 = cp.dot(W3.T, dZ3) * (Z2 > 0)
    dW2 = cp.dot(dZ2, A1.T) / m
    db2 = cp.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = cp.dot(W2.T, dZ2) * (Z1 > 0)
    dW1 = cp.dot(dZ1, X.T) / m
    db1 = cp.sum(dZ1, axis=1, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, dW6, db6


# update parameters
def update_parameters(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, dW6, db6, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3
    W4 = W4 - learning_rate * dW4
    b4 = b4 - learning_rate * db4
    W5 = W5 - learning_rate * dW5
    b5 = b5 - learning_rate * db5
    W6 = W6 - learning_rate * dW6
    b6 = b6 - learning_rate * db6

    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6

# compute lost
def compute_lost(A4, Y):
    m = Y.shape[1]
    lost = -cp.sum(Y * cp.log(A4)) / m
    return lost

# compute accuracy
def compute_accuracy(A4, Y):
    m = Y.shape[1]
    p = cp.argmax(A4, axis=0)
    q = cp.argmax(Y, axis=0)
    accuracy = cp.sum(p == q) / m
    return accuracy


def one_hot_labels(labels):
    m = labels.shape[0]
    n = 10
    one_hot = cp.zeros((m, n))
    one_hot[cp.arange(m), labels] = 1
    return one_hot

if __name__ == '__main__':

    # read MNIST label data
    magic, num_labels = read_label_headers('./data/train-labels.idx1-ubyte')
    labels = read_label_data('train-labels.idx1-ubyte')
    print('MNIST label data')
    print('magic number: {}'.format(magic))
    print('number of labels: {}'.format(num_labels))
    print()

    # read MNIST image data
    magic, num_images, num_rows, num_cols = read_image_headers('./data/train-images.idx3-ubyte')
    images = read_image_data('train-images.idx3-ubyte')
    print('MNIST image data')
    print('magic number: {}'.format(magic))
    print('number of images: {}'.format(num_images))
    print('shape of images matrices: ', cp.shape(images))

    # 5 layers, 4 hidden layers neural network, output layer size 10
    # input layer size 784
    # hidden layer size 128

    # initialize weights and biases
    NODE_NUM = 128

    W1 = cp.random.rand(NODE_NUM, 784)-0.5
    b1 = cp.random.rand(NODE_NUM, 1)-0.5
    W2 = cp.random.rand(NODE_NUM, NODE_NUM)-0.5
    b2 = cp.random.rand(NODE_NUM, 1)-0.5
    W3 = cp.random.rand(NODE_NUM, NODE_NUM)-0.5
    b3 = cp.random.rand(NODE_NUM, 1)-0.5
    W4 = cp.random.rand(NODE_NUM, NODE_NUM)-0.5
    b4 = cp.random.rand(NODE_NUM, 1)-0.5
    W5 = cp.random.rand(NODE_NUM, NODE_NUM)-0.5
    b5 = cp.random.rand(NODE_NUM, 1)-0.5
    W6 = cp.random.rand(10, NODE_NUM)-0.5
    b6 = cp.random.rand(10, 1)-0.5

    # print b4
    for i in range(10):
        print(b4[i])

    # learning rate
    alpha = 0.0001

    X = (images/255).T
    Y = one_hot_labels(labels).T

    # training
    time_start = time.time()
    for i in range(100):
        Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5, Z6, A6 \
            = forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6)
        dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, dW6, db6 \
            = backward_propagation(X, Y, Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5, Z6, A6, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6)
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6 \
            = update_parameters(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, dW6, db6, alpha)

    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    print('\n**********************\n')


    # test
    # read MNIST label data from test data
    magic, num_labels = read_label_headers('./data/t10k-labels.idx1-ubyte')
    labels = read_label_data('t10k-labels.idx1-ubyte')
    print('MNIST label data')
    print('magic number: {}'.format(magic))
    print('number of labels: {}'.format(num_labels))
    print('labels: {}'.format(labels))
    print()

    # read MNIST image data from test data
    magic, num_images, num_rows, num_cols = read_image_headers('./data/t10k-images.idx3-ubyte')
    images = read_image_data('t10k-images.idx3-ubyte')
    print('MNIST test data')
    print('magic number: {}'.format(magic))
    print('number of images: {}'.format(num_images))
    print('shape of images matrices: ', cp.shape(images))

    # pre-process image
    images = images / 255
    X = images.T

    # load weight and biases from weights.txt and biases.txt (space separated)

    # test
    correct = 0

    Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5, Z6, A6 = forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6)

    for i in range(10000):
        if cp.argmax(A6[:, i]) == labels[i]:
            correct += 1

    print('correct: {}'.format(correct))
    print("accuracy: ", correct / 10000)





