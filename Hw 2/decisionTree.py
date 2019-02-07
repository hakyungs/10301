from __future__ import division
import sys
import math
import copy


'''
LIST OF TOP LEVEL (FINAL) VARS:
    trainData - original 2d list training data
    att_list - list of names of attributes
    att_indicators - list of "true" values for each attributes
    att_non_indicators - list of "false" values for each attributes
    max_depth - maximum number of splits
    DT - Decision tree created in main method
'''


class DT_Node(object):
    def __init__(self,td,att):
        self.att = att
        self.left = None
        self.right = None
        self.posNeg = getPosNegY(td)
        self.numPos = len(self.posNeg[0])
        self.numNeg = len(self.posNeg[1])

    def pretty_print(self):
        print("[%d %s /%d %s]" % (self.numPos,att_indicators[-1],
                                  self.numNeg,att_non_indicators[-1]))
        def helper(self,d,att_pass,dir):
            if (dir == "l"):
                print("|  " * d + "%s = %s: [%d %s /%d %s]" %
                      (att_list[att_pass], att_indicators[att_pass],
                       self.numPos,att_indicators[-1],
                       self.numNeg,att_non_indicators[-1]))
            elif (dir == "r"):
                print("|  " * d + "%s = %s: [%d %s /%d %s]" %
                      (att_list[att_pass], att_non_indicators[att_pass],
                       self.numPos,att_indicators[-1],
                       self.numNeg,att_non_indicators[-1]))
            # stop if we're at a leaf, else recurse
            if (self.left != None): helper(self.left,d+1,self.att,"l")
            if (self.right != None): helper(self.right,d+1,self.att,"r")

        helper(self.left,1,self.att,"l")
        helper(self.right,1,self.att,"r")

        return None

    def getDepth() :
        if (self.left == None and self.right == None): return 0
        elif (self.left == None) : return (self.right.getDepth()+1)
        elif (self.right == None): return (self.left.getDepth()+1)
        else: return max(self.left.getDepth()+1,self.right.getDepth()+1)

'''
//TESTED//
getPosNegY : filters out negative Y entries for given attribute and returns
tuple of data 2d list where (i,j)
i = only positive entries
j = only negative entries
L : data to edit
'''

def getPosNegY(L):
    negs = []
    poss = []
    for entry in L:
        if (entry[-1] != att_indicators[-1]):
            negs.append(entry)
        else:
            poss.append(entry)
    return (poss,negs)


'''
//TESTED//
getPosNegAtt : filters out negative indicator entries for given attribute and returns
tuple of data 2d list where (i,j)
i = only positive entries
j = only negative entries
L : data to edit
att : attribute INDEX number to pick out positive values from
'''
def getPosNegAtt(L,att):
    negs = []
    poss = []
    for entry in L:
        if (entry[att] != att_indicators[att]):
            negs.append(entry)
        else:
            poss.append(entry)
    return (poss,negs)

'''
//TESTED//
get_MI : returns mutual information given by attribute att_list[att]
td = dataset to use for calculation
att = INDEX of attribute
'''
def get_MI(td,att):
    # H(Y) = -[P(Y=0)log_2(P(Y=0)) + P(Y=1)log_2(P(Y=1))]
    totalObs = len(td)
    tdPos = []
    tdNeg = []
    for entry in td:
        if (entry[-1] == att_indicators[-1]):
            tdPos.append(entry)
        else: tdNeg.append(entry)
    if (totalObs > 0):
        py0 = len(tdNeg) / totalObs #P(Y=0)
        py1 = len(tdPos) / totalObs #P(Y=1)
    else:
        py0 = 0
        py1 = 0
    if (py0 == 0 and py1 == 0): entY = 0
    elif (py0 == 0): entY =  -1 * (py1 * math.log(py1,2))
    elif (py1 == 0): entY = -1 * (py0 * math.log(py0,2))
    else:
        entY = -1 * ((py0 * math.log(py0,2)) + (py1 * math.log(py1,2))) # H(Y)
    # H(Y|att) = P(att=0)H(Y|att=0) + P(att=1)H(Y|att=1)
    atPos = []
    atNeg = []
    for entry in td:
        if (entry[att] == att_indicators[att]):
            atPos.append(entry)
        else: atNeg.append(entry)
    if (totalObs > 0):
        at0 = len(atNeg) / totalObs #P(att=0)
        at1 = len(atPos) / totalObs #P(att=1)
    else:
        at0 = 0
        at1 = 0
    # H(Y|att=0) = -[P(Y=0|att=0)log_2(P(Y=0|att=0)) + P(Y=1|att=0)log_2(P(Y=1|att=0))]
    y1a0_count=0
    for entry in atNeg:
        if (entry[-1] == att_indicators[-1]):
            y1a0_count += 1
    y0a0_count = len(atNeg) - y1a0_count
    if (at0 > 0):
        py0a0 = (y0a0_count / totalObs) / at0 #P(Y=0|att=0)
        py1a0 = (y1a0_count / totalObs) / at0 #P(Y=1|att=0)
    else:
        py0a0 = 0
        py1a0 = 0
    if (py0a0 == 0 and py1a0 == 0): entYA0 = 0
    elif (py0a0 == 0): entYA0 = -1 * (py1a0 * math.log(py1a0,2))
    elif (py1a0 == 0): entYA0 = -1 * (py0a0 * math.log(py0a0,2))
    else:
        entYA0 = -1 * ((py0a0 * math.log(py0a0,2)) + (py1a0 * math.log(py1a0,2))) #H(Y|att=0)
    # H(Y|att=1) = -[P(Y=0|att=1)log_2(P(Y=0|att=1)) + P(Y=1|att=1)log_2(P(Y=1|att=1))]
    y1a1_count=0
    for entry in atPos:
        if (entry[-1] == att_indicators[-1]):
            y1a1_count += 1
    y0a1_count = len(atPos) - y1a1_count
    if (at1 > 0):
        py0a1 = (y0a1_count / totalObs) / at1 #P(Y=0|att=1)
        py1a1 = (y1a1_count / totalObs) / at1 #P(Y=1|att=1)
    else:
        py0a1 = 0
        py1a1 = 0
    if (py0a1 == 0 and py1a1 == 0): entYA1 = 0
    elif (py0a1 == 0): entYA1 = -1 * (py1a1 * math.log(py1a1,2))
    elif (py1a1 == 0): entYA1 = -1 * (py0a1 * math.log(py0a1,2))
    else:
        entYA1 = -1 * ((py0a1 * math.log(py0a1,2)) + (py1a1 * math.log(py1a1,2))) #H(Y|att=1)
    # H(Y|att)
    entYA = (at0 * entYA0) + (at1 * entYA1)
    return (entY - entYA)


'''
//TESTED//
create_DT : returns DT (DT_Node class) for given trainData
trainData : 2d list [[att1,att2,...],....]
att_list : 1d list of remaining possible attributes' indices [0,1,2,...]
max_depth : maximum number of split nodes
'''

def create_DT(td,al,curr_depth):
    # Leaf node
    if (td == []): return DT_Node(td,None)
    if ((curr_depth >= max_depth) or (al == [])) :
        return DT_Node(td,None)
    # pick which attribute to split on
    else:
        maxMI = 0
        split_att = al[0]
        for att in al:
            currMI = get_MI(td,att)
            if (currMI > maxMI):
                maxMI = currMI
                split_att = att
        # only split if mutual information > 0
        if (maxMI <= 0): return (DT_Node(td,None))
        root =  DT_Node(td,split_att)
        divData = getPosNegAtt(td,split_att)
        alcpy = copy.deepcopy(al)
        alcpy.remove(split_att)
        root.left = create_DT(divData[0],alcpy,curr_depth+1)
        root.right = create_DT(divData[1],alcpy,curr_depth+1)
        return root


'''
traverse : given attribute inputs, follow the DT and output the prediction
entry : list of attributes + actual result at entry[-1]
usage: if (traverse(entry) != entry[-1]):
'''

def traverse(l):
    attrs = l[:-1]



if __name__ == "__main__" :
    i1 = sys.argv[1]
    i2 = sys.argv[2]
    i3 = sys.argv[3]
    i4 = sys.argv[4]
    i5 = sys.argv[5]
    i6 = sys.argv[6]

    train_input = open(i1,"r")
    test_input = open(i2,"r")
    max_depth = int(i3)
    train_out = open(i4,"w")
    test_out = open(i5,"w")
    metrics_out = open(i6,"w")

    data = train_input.read()
    test_raw = test_input.read()

    ####################################################################
    # Learning
    ####################################################################

    # format data into 2d list : [[att1,att2,att3,....,y],...]
    # saves list of names of attributes : att_list = [att1,att2,...]
    lines = data.splitlines()
    trainData = [-1 for i in range(len(lines)-1)]
    for i in range(len(lines)):
        if (i==0): att_list = lines[i].split(",")
        else: trainData[i-1] = lines[i].split(",")


    # first value of each attribute becomes "positive" (= indicator) value!!!!
    att_indicators = trainData[0]
    att_non_indicators = [-1 for i in range(len(att_indicators))]
    for entry in trainData:
        for j in range(len(att_indicators)):
            if (entry[j] != att_indicators[j]):
                att_non_indicators[j] = entry[j]
    # len(att_list) since all attributes are available at start
    DT = create_DT(trainData,[i for i in range(len(att_list)-1)],0)
    DT.pretty_print()


    ####################################################################
    # Test on training data
    ####################################################################

    train_error_count = 0
    for entry in trainData:
        if (traverse(entry) != entry[-1]):
            train_error_count += 1
    train_error = train_error_count / len(trainData)

    ####################################################################
    # Test on testing data
    ####################################################################

    test_lines = test_raw.splitlines()
    testData = [-1 for i in range(len(test_lines)-1)]
    for i in range(len(test_lines)):
        if (i==0): continue
        else: testData[i-1] = test_lines[i].split(",")

    test_error_count = 0
    for entry in testData:
        if (traverse(entry) != entry[-1]):
            test_error_count += 1
    test_error = test_error_count / len(testData)



    train_input.close()
    test_input.close()
    train_out.close()
    test_out.close()
    metrics_out.close()
