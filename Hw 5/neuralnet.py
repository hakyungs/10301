from __future__ import division
import sys
import math
import numpy as np


#########################################################################
# Formatting Data
#########################################################################
def formatData(d):
    res = []
    for line in d.split("\n"):
        splitted = line.split(",")
        if (splitted[0] == ''): continue
        label = int(splitted[0])
        features = splitted[1:]
        l = [0 for i in range(10)]
        l[label] = 1
        featureArr = np.array([features])
        featureArr = featureArr.astype(np.float)
        res.append((featureArr,np.array(l))) #tuple of feature & label

    return np.array(res)

#########################################################################
# Training
#########################################################################

def sigmoid(x):
	return (1 / (1 + np.exp(-x)))

def NNForward(x, y, alpha, beta):
    a = np.dot(alpha,x.T) # linearForward(alpha,x.T) - scalar!!
    z = np.insert(sigmoid(a),0,1)
    z = (np.array([z])).T
    b = np.dot(beta,z) # linearForward(beta,z)
    yHat = np.exp(b) / (np.sum(np.exp(b))) # softmaxForward(b)
    J = -1 * (np.dot(np.array(y), np.log(yHat))) #crossEntropyForward(y,yHat)
    o = (a,z,b,yHat,J)
    return o


def NNBackward(x, y, alpha, beta, o):
    (a,z,b,yHat,J) = o
    y = (np.array([y])).T
    gb = yHat - y # softmaxBackward(b,yHat,gy) 10 x 1
    gBeta = np.dot(gb,z.T) # linearBackward(z,b,gb) 10 x 5

    beta = np.delete(beta,0,1)
    gz = np.dot(beta.T, gb) # linearBackward(z,b,gb) 4 x 1
    z = np.delete(z,0,0)
    ga = gz * z * (1-z) #sigmoidBackward(a,z,gz) 4 x 1
    gAlpha = np.dot(ga,x) #linearBackward(x,a,ga) 4 x 129
    return (gAlpha, gBeta)


#########################################################################
# Testing
#########################################################################

def evalMCE(data_list,alpha,beta):
    crossEntropy = 0.0
    prediction_list = []
    for d in data_list:
        x = np.array([np.insert(d[0],0,1)])
        y = d[1]
        (a,z,b,yHat,J) = NNForward(x,y,alpha,beta)
        expected_outcome = np.argmax(yHat)
        prediction_list.append(expected_outcome)
        crossEntropy += J
    lenn = data_list.shape[0]
    MCE = crossEntropy / lenn
    return (MCE,prediction_list)

#########################################################################
# Main Function
#########################################################################
if __name__ == "__main__" :
    i1 = sys.argv[1]
    i2 = sys.argv[2]
    i3 = sys.argv[3]
    i4 = sys.argv[4]
    i5 = sys.argv[5]
    i6 = sys.argv[6]
    i7 = sys.argv[7]
    i8 = sys.argv[8]
    i9 = sys.argv[9]

    train_input = open(i1,"r")
    test_input = open(i2,"r")
    train_out = open(i3,"w")
    test_out = open(i4,"w")
    metrics_out = open(i5,"w")
    num_epoch = int(i6)
    hidden_units = int(i7)
    init_flag = int(i8)
    learning_rate = float(i9)

#########################################################################
#   Format given inputs
#########################################################################
    train_raw = train_input.read()
    test_raw = test_input.read()
    train_data = formatData(train_raw)
    test_data = formatData(test_raw)

    num_feature = train_data[0][0].size

#########################################################################
#   Initialize according to init_flag
#########################################################################
    alpha, beta = 0.0,0.0

    # init randomly, bias set to 0
    if (init_flag == 1) :
        alpha = np.random.uniform(-0.1,0.1,(hidden_units,num_feature+1))
        beta = np.random.uniform(-0.1,0.1,(10,hidden_units+1))
        #beta = np.random.rand(10, hidden_units+1)
        #beta = (beta * 0.2) - 0.1
        #for l in beta : l[0] = 0

    else :
        alpha = np.zeros((hidden_units, num_feature+1))
        beta = np.zeros((10, hidden_units+1))

#########################################################################
# Perform SGD
#########################################################################

    for i in range(num_epoch):
        for d in train_data:
            x = np.array([np.insert(d[0],0,1)])
            y = d[1]
            o = NNForward(x,y,alpha,beta)
            (gAlpha,gBeta) = NNBackward(x,y,alpha,beta,o)
            alpha -= gAlpha * learning_rate
            beta -= gBeta * learning_rate

        (train_MCE,train_exp) = evalMCE(train_data,alpha,beta)
        (test_MCE,test_exp) = evalMCE(test_data,alpha,beta)

        metrics_out.write("epoch=%d crossentropy(train): %f" % (i+1,train_MCE)
                          + "\n")
        metrics_out.write("epoch=%d crossentropy(test): %f" % (i+1,test_MCE)
                          + "\n")

    wrongCount = 0
    for i in range(len(train_exp)):
        if (train_exp[i] != np.argmax(train_data[i][1])): wrongCount += 1
        train_out.write("%d" % train_exp[i] + "\n")

    train_error = wrongCount / len(train_exp)

    wrongCount = 0
    for i in range(len(test_exp)):
        if (test_exp[i] != np.argmax(test_data[i][1])): wrongCount += 1
        test_out.write("%d" % test_exp[i] + "\n")

    test_error = wrongCount / len(test_exp)

    metrics_out.write("error(train): %f" % train_error + "\n")
    metrics_out.write("error(test): %f" % test_error + "\n")

    train_input.close()
    test_input.close()
    train_out.close()
    test_out.close()
    metrics_out.close()
