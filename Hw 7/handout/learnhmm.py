from __future__ import division
import sys
import math
import numpy as np

#########################################################################
# Formatting Data
#########################################################################
def formatData(d,data_type):
    if (data_type == words):
        res = []
        for line in d.readlines():
			tokens = line.strip().split(" ")
			token_list = []
			for t in tokens:
				token = t.split("_")
				token_list.append(token)
			res.append(token_list)
    return

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

    train_input = open(i1,"r")
    index_to_word = open(i2,"r")
    index_to_tag = open(i3,"r")
    hmmprior = open(i4,"w")
    hmmemit = open(i5,"w")
    hmmtrans = open(i6,"w")

    train_data = formatData(train_input,words)
    word_idx_data = formatData(index_to_word,idx)
    tag_idx_data = formatData(index_to_tag,idx)

    train_input.close()
    index_to_word.close()
    index_to_tag.close()
    hmmprior.close()
    hmmemit.close()
    hmmtrans.close()
