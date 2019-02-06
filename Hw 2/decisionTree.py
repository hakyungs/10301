import sys
import math

'''
LIST OF TOP LEVEL (FINAL) VARS:
    trainData - original 2d list training data
    att_list - list of names of attributes
    att_indicators - list of "true" values for each attributes
    max_depth - maximum number of splits
'''



class DT_Node(object):
    def __init__(self,td,att):
        self.att = att
        self.left = None
        self.right = None
        self.numPos =
        self.numNeg =
    def pretty_print(self):
        return None
    def getDepth() :
        return max(self.left.getDepth()+1,self.right.getDepth()+1)

'''
getPosNeg : filters out negative indicator entries for given attribute and returns
tuple of data 2d list where (i,j)
i = only positive entries
j = only negative entries
L : data to edit
att : attribute INDEX number to pick out positive values from
'''
def getPosNeg(L,att):
    negs = []
    for entry in L:
        if (entry[att] != att_indicators[att]):
            negs.append(entry)
            L.remove(entry)
    return (L,negs)

'''
create_DT : returns DT (DT_Node class) for given trainData
trainData : 2d list [[att1,att2,...],....]
att_list : 1d list of remaining possible attributes' indices [0,1,2,...]
max_depth : maximum number of split nodes
'''

def create_DT(td,al,curr_depth):
    # Leaf node
    if ((curr_depth >= max_depth) or (al == [])) :
        return DT_Node(td,None)
    # pick which attribute to split on
    else:
        maxMI = 0
        split_att = al[0]
        for att in al:
            currMI = get_MI(att)
            if (currMI > maxMI):
                maxMI = currMI
                split_att = att
        # only split if mutual information > 0
        if (maxMI <= 0): return DT_Node(td,None)
        root =  DT_Node(td,split_att)
        divData = getPosNeg(td,split_att)
        new_split_att = al.remove(split_att)
        root.left = create_DT(divData[0]),new_split_att,curr_depth+1)
        root.right = create_DT(divData[1],new_split_att,curr_depth+1)
        return root







if __name__ == ’__main__’ :
    i1 = sys.argv[1]
    i2 = sys.argv[2]
    i3 = sys.argv[3]
    i4 = sys.argv[4]
    i5 = sys.argv[5]
    i6 = sys.argv[6]

    train_input = open(i1,"r")
    test_input = open(i2,"r")
    max_depth = i3
    train_out = open(i4,"w")
    test_out = open(i5,"w")
    metrics_out = open(i6,"w")

    data = train_input.read()

    # format data into 2d list : [[att1,att2,att3,....,y],...]
    # saves list of names of attributes : att_list = [att1,att2,...]
    lines = data.splitlines()
    trainData = []
    for i in range(len(lines)):
        if (i==0): att_list = lines[i].split(",")
        else: trainData[i-1] = lines[i].split(",")

    # first value of each attribute becomes "positive" (= indicator) value!!!!
    att_indicators = trainData[0]
    # len(att_list) since all attributes are available at start
    DT = create_DT(trainData,[i for i in range(len(att_list))],0,max_depth)
    DT.pretty_print()



    train_input.close()
    test_input.close()
    train_out.close()
    test_out.close()
    metrics_out.close()
