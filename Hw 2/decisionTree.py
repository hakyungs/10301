import sys
import math

class DT_Node(object):
    def __init__(self,att,depth):
        self.att = att
        self.depth = depth
        self.left = None
        self.right = None
    def pretty_print(self):


class DT_Leaf(object):
    def __init__(self,):
        self.pPos =
        self.pNeg =

'''
trainData : 2d list [[att1,att2,...],....]
att_list : 1d list of remaining possible attributes' indices [0,1,2,...]
returns DT with trainData
'''
def create_DT(trainData,att_list):
    maxMI = 0
    split_att = att_list[0]
    for att in att_list:
        currMI = get_MI(att)
        if (currMI > maxMI):
            maxMI = currMI
            split_att = att
    root =  DT_Node(split_att,0)
    root.left = create_DT()
    root.right = create_DT()
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
        if (i==0): att_list = lines[i]
        else: trainData[i-1] = lines[i].split(",")

    # len(att_list) since all attributes are available at start
    DT = create_DT(trainData,[i for i in range(len(att_list))])
    DT.pretty_print()

    

    train_input.close()
    test_input.close()
    train_out.close()
    test_out.close()
    metrics_out.close()
