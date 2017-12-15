import numpy as np
from numpy import vectorize
from loadData import train_x, train_y, test_x, test_y

# load data
trainData = np.transpose(train_x)
trainLabel = np.transpose(train_y)
testData = np.transpose(test_x)
testLabel = np.transpose(test_y)
dataSetSize = trainLabel.size
# parameters
dimension = 100
layer1Size = 50
layer2Size = 10
outputSize = 1
Beta = 10
Gamma = 1
growRate = 5
warmStart = 1
threshold = 1e-4
# store the result
accList = []
lossList = []
# Initialize the network
a0 = trainData
a0_pinv = np.linalg.pinv(a0)
W1 = np.zeros((layer1Size,dimension))
var = 1
z1 = var * np.random.randn(layer1Size,dataSetSize)
a1 = var * np.random.randn(layer1Size,dataSetSize)
W2 = np.zeros((layer2Size,layer1Size))
z2 = var * np.random.randn(layer2Size,dataSetSize)
a2 = var * np.random.randn(layer1Size,dataSetSize)
W3 = np.zeros((outputSize,layer2Size))
z3 = var * np.random.randn(1,dataSetSize)
_lambda = np.zeros((1,dataSetSize))

def activation(i):
    if i > 0:
        return i
    else:
        return 0

def get_z_l(a,w_a):
    def f_z(z):
        return Gamma*(a-activation(z))**2 + Beta*(z-w_a)**2

    z1 = max((a*Gamma+w_a*Beta)/(Beta+Gamma),0)
    result1 = f_z(z1)

    z2 = min(w_a,0)
    result2 = f_z(z2)

    if result1 <= result2:
        return z1
    else:
        return z2

def get_z_L(y,w_a,_lambda):
    if y==-1:
        def f_z(z):
            return Beta*z**2 - (2*Beta*w_a-_lambda)*z + max(1+z,0)
        z1 = min((2*Beta*w_a - _lambda)/(2*Beta),-1)
        z2 = max((2*Beta*w_a-_lambda-1)/(2*Beta),-1)
        if f_z(z1) < f_z(z2):
            return z1
        else:
            return z2
    if y==1:
        def f_z(z):
            return Beta*z**2 - (2*Beta*w_a - _lambda)*z + max(1-z,0)
        z1 = min((2*Beta*w_a - _lambda+1)/(2*Beta),1)
        z2 = max((2*Beta*w_a - _lambda)/(2*Beta),1)

        if f_z(z1) < f_z(z2):
            return z1
        else:
            return z2

    else:
        print("error class: {}".format(y))
        exit()

def get_predict(pre):
    if pre >= 0:
        return 1
    else:
        return -1

def get_loss(pre,gt):
    if gt==-1:
        return max(1+pre,0)
    elif gt==1:
        return max(1-pre,0)
    else:
        print("invalid gt..")
        exit()

vactivation = vectorize(activation)
vget_z_l = vectorize(get_z_l)
vget_z_L = vectorize(get_z_L)
vget_predict = vectorize(get_predict)
vget_loss = vectorize(get_loss)

def update(is_warmStart = False):
    global  z1,z2,z3,_lambda,W1,W2,W3
    # update layer1
    old_W1 = W1
    old_z1 =z1
    W1 = np.dot(z1, a0_pinv)
    a1_left = np.linalg.inv((Beta * np.dot(np.transpose(W2), W2) + Gamma * np.eye(layer1Size, dtype=float)))
    a1_right = (Beta * np.dot(np.transpose(W2), z2) + Gamma * vactivation(z1))
    a1 = np.dot(a1_left,a1_right)
    z1 = vget_z_l(a1, np.dot(W1, a0))

    # update layer
    W2 = np.dot(z2, np.linalg.pinv(a1))
    a2_left = np.linalg.inv((Beta * np.dot(np.transpose(W3), W3) + Gamma * np.eye(layer2Size, dtype=float)))
    a2_right = (Beta * np.dot(np.transpose(W3), z3) + Gamma * vactivation(z2))
    a2 = np.dot(a2_left , a2_right)
    z2 = vget_z_l(a2, np.dot(W2, a1))

    # update last layer
    W3 = np.dot(z3, np.linalg.pinv(a2))
    z3 = vget_z_L(trainLabel, np.dot(W3, a2),_lambda)

    loss = vget_loss(z3,trainLabel)
    if not is_warmStart:
        _lambda = _lambda + Beta * (z3 - np.dot(W3,a2))
    # mean square error
    mse = np.linalg.norm(old_z1 - z1, 2)
    return mse

def test_admm():
    a0 = testData
    layer_1_output = vactivation(np.dot(W1,a0))
    layer_2_output = vactivation(np.dot(W2,layer_1_output))
    predict = np.dot(W3,layer_2_output)
    pre = vget_predict(predict)
    hit = np.equal(pre,testLabel)
    acc = np.sum(hit)*1.0/testLabel.size
    accList.append(acc)
    print("test data predict accuracy: {}".format(acc))


def train_and_test_admm():
    global  Beta,Gamma
    for i in range(warmStart):
        loss = update(is_warmStart=True)
        print("warm start, err :{}".format(loss))
    iteration = 1
    while 1:
        loss = update(is_warmStart=False)
        lossList.append(loss)
        print("iteration {}, err :{}".format(i,loss))
        test_admm()
        iteration = iteration + 1
        if loss < threshold:
            break

if __name__ == '__main__':
    train_and_test_admm()
    with open('loss.txt', 'w') as f:
        for e in lossList:
            f.write(str(e)+'\n')
    with open('acc.txt', 'w') as f:
        for e in accList:
            f.write(str(e)+'\n')



