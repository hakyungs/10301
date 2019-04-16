from __future__ import division
import sys
import math
import numpy as np

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

    test_input_raw = open(i1,"r")
    index_to_word_raw = open(i2,"r")
    index_to_tag_raw = open(i3,"r")
    hmmprior_out = open(i4,"r")
    hmmemit_out = open(i5,"r")
    hmmtrans_out = open(i6,"r")
    predicted_file = open(i7,"w")
    metric_file = open(i8,"w")



    test_input_raw.close()
    index_to_word_raw.close()
    index_to_tag_raw.close()
    hmmprior_out.close()
    hmmemit_out.close()
    hmmtrans_out.close()
    predicted_file.close()
    metric_file.close()
