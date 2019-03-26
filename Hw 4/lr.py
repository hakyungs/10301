from __future__ import division
import sys
import math


#########################################################################
# dot(theta,xi)
# len(theta) = len(dict)
# returns dot product
#########################################################################

def dot(v1,v2):
    res = v1[0] #bias term
    for idx in v2:
        res += v1[idx+1]
    return res

def onlyIndex(xi):
    for i in range(len(xi)):
        jthFeature = xi[i]
        idx = int(jthFeature.split(":")[0])
        xi[i] = idx
    return xi

#########################################################################
# SGD(training_ex, theta)
# Param : theta(=model param), training_ex
# returns : updated theta
# takes a single SGD step on the ith training example. Updates the model
# parameters in place by taking one stochastic gradient step.
#########################################################################

def sgd(training_ex,theta):
    learning_rate = 0.1
    splited = training_ex.split("\t") #[yi,x1,x2,x3,...,xN]
    yi = int(splited[0])
    xi = splited[1:-1]
    xi = onlyIndex(xi)
    dotProd = dot(theta,xi)
    nxj = learning_rate
    eDotProd = math.exp(dotProd)
    denom = 1 + (eDotProd)
    theta[0] += nxj * (yi - (eDotProd / denom)) #bias
    for idx in xi:
        theta[idx+1] += nxj * (yi - (eDotProd / denom))
    return theta

#########################################################################
# format_dict(dict_raw)
# Takes in raw dict txt file and converts to actual dictionary type
# param : dict_raw
# returns : dictionary with words as keys and index as value
#########################################################################
def format_dict(data):
    res = dict()
    for line in data.splitlines():
        splited = line.split(" ")
        word,idx = splited[0],splited[1]
        res[word] = idx
    return res

#########################################################################
# predict(xi,theta)
#########################################################################
'''
def predict(xi,theta):
    u = dot(theta,xi)
    p = 1 / (1 + math.exp(-u))
    if (p > 0.5): return 1
    else : return 0
'''

def predict(xi,theta):
    u = dot(theta,xi)
    if (u > 0): return 1
    else: return 0

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

    train_input = open(i1,"r")
    validation_input = open(i2,"r")
    test_input = open(i3,"r")
    dict_input = open(i4,"r")
    train_out = open(i5,"w")
    test_out = open(i6,"w")
    metrics_out = open(i7,"w")
    num_epoch = int(i8)

    train_data = train_input.read()
    validation_data = validation_input.read()
    test_data = test_input.read()

    train_data = train_data.splitlines()
    validation_data = validation_data.splitlines()
    test_data = test_data.splitlines()

    dict_raw = dict_input.read()
    word_dict = format_dict(dict_raw)

####################################################################
# Learning
####################################################################
    #Train
    theta = [0 for i in range(len(word_dict)+1)]
    for epoch in range(num_epoch):
        for ex in train_data:
            theta = sgd(ex,theta)

####################################################################
# Testing
####################################################################
    wrongCount = 0
    for ex in train_data:
        splited = ex.split("\t") #[yi,x1,x2,x3,...,xN]
        yi = int(splited[0])
        xi = splited[1:-1]
        print(xi)
        xi = onlyIndex(xi)
        expected_outcome = predict(xi,theta)
        actual_outcome = yi
        train_out.write("%d" % expected_outcome + "\n")
        if (expected_outcome != actual_outcome): wrongCount += 1
    train_error = wrongCount / len(train_data)

    wrongCount = 0
    for ex in test_data:
        splited = ex.split("\t") #[yi,x1,x2,x3,...,xN]
        yi = int(splited[0])
        xi = splited[1:-1]
        xi = onlyIndex(xi)
        expected_outcome = predict(xi,theta)
        actual_outcome = yi
        test_out.write("%d" % expected_outcome + "\n")
        if (expected_outcome != actual_outcome): wrongCount += 1
    test_error = wrongCount / len(test_data)

    metrics_out.write("error(train): %f" % train_error + "\n")
    metrics_out.write("error(test): %f" % test_error + "\n")

    train_input.close()
    validation_input.close()
    test_input.close()
    dict_input.close()
    train_out.close()
    test_out.close()
    metrics_out.close()
