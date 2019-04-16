from __future__ import division
import sys
import math
import numpy as np

#########################################################################
# Formatting Data
#########################################################################


def formatData(d,data_type):
    # train_input
    if (data_type == 0):
        res = []
        for line in d.readlines():
            tokens = line.strip().split(" ")
            token_list = []
            for t in tokens:
                token = t.split("_")
                token_list.append(token)
            res.append(token_list)
    elif (data_type == 1):  # index_to_tag & index_to_word
        res = {}
        row = []
        for line in d.readlines():
            if (line.strip() != ""):
                row.append(line.split()[0])
            for i in range(len(row)):
                res[row[i]] = i

    return res


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

    train_input_raw = open(i1,"r")
    index_to_word_raw = open(i2,"r")
    index_to_tag_raw = open(i3,"r")
    hmmprior_out = open(i4,"w")
    hmmemit_out = open(i5,"w")
    hmmtrans_out = open(i6,"w")

    train_input = formatData(train_input_raw,0)
    index_to_word = formatData(index_to_word_raw,1)
    index_to_tag = formatData(index_to_tag_raw,1)

    w_len = len(index_to_word)
    t_len = len(index_to_tag)

    # initialize np arrays
    hmmtrans = np.zeros((t_len,t_len))
    hmmemit = np.zeros((t_len,w_len))
    hmmprior = np.zeros(t_len)

    # fill the matrices
    for sentence in train_input:
        prev = None
        for i, pair in enumerate(sentence):
            word_idx,tag_idx = pair[0],pair[1]
            word,tag = index_to_word[word_idx],index_to_tag[tag_idx]
            if (i == 0) :
                hmmprior[tag] += 1
            else :
                hmmtrans[prev][tag] += 1
            prev = tag
            hmmemit[tag][word] += 1

    hmmprior += 1
    hmmtrans += 1
    hmmemit += 1

    #normalize
    for row in hmmtrans:
        row /= np.sum(row)
    for row in hmmemit:
        row /= np.sum(row)
    hmmprior /= np.sum(hmmprior)


    # write outputs
    for row in hmmtrans:
        toPrint = ""
        for entry in row:
            toNum = "%.18e"%entry
            toPrint += " " + toNum
        hmmtrans_out.write(toPrint + "\n")


    train_input_raw.close()
    index_to_word_raw.close()
    index_to_tag_raw.close()
    hmmprior_out.close()
    hmmemit_out.close()
    hmmtrans_out.close()
