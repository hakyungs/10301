from __future__ import division
import sys
import math
import numpy as np

#########################################################################
# Formatting Data
#########################################################################
def formatData(d,data_type):

    if (data_type == 1):  # index_to_tag & index_to_word
        res_0 = {}
        res_inv = {} #tag_to_idx & word_to_idx
        row = []
        for line in d.readlines():
            if (line.strip() != ""):
                row.append(line.split()[0])
        for i in range(len(row)):
            res_0[i] = row[i]
            res_inv[row[i]] = i
        res = (res_0,res_inv)

    elif (data_type == 2): #hmmprior
        tmp = d.readlines()
        res = np.zeros(len(tmp))
        for i,line in enumerate(tmp):
            string_d = line[:-1] #strip \n
            res[i] = float(string_d)

    elif (data_type == 3): #hmmemit & hmmtrans
        res = []
        tmp = d.readlines()
        for row in tmp:
            cur_str = row.strip("\n")
            str_list = cur_str.split()
            new_l = []
            for col in range(len(str_list)):
                new_l.append(float(str_list[col]))
            res.append(np.array(new_l))
        res = np.array(res)

    return res

#########################################################################
# Get Alpha & Beta
#########################################################################
def getAlpha(sentence,pi,A,B):
    alpha = []
    log_likelihood_list = []
    for t,pair in enumerate(sentence):
        word_index = pair[0]
        b_jx = (B.T)[word_index]
        if (t==0):
            a_tj = pi.T * b_jx.T
        else :
            a_tj = b_jx.T * np.dot(B.T, alpha[t-1].T)
        alpha.append(a_tj / np.sum(a_tj))

        #update log Likelihood
        if (t==len(sentence)-1):
            curr_log_likelihood = np.log(np.sum(a_tj))
            log_likelihood_list.append(curr_log_likelihood)
    return alpha,log_likelihood_list

def getBeta(sentence,pi,A,B):
    beta = []
    for t in range(len(seq)-1,0,-1):
        if (t==0):
            beta.append(np.array([1] * len(pi))) #check
        else :
            word_index = sentence[t][0]
            b_tj = (B.T)[word]
            beta = [np.dot(A,beta[0].T * b_tj.T)] + beta
    return beta

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
    hmmprior_raw = open(i4,"r")
    hmmemit_raw = open(i5,"r")
    hmmtrans_raw = open(i6,"r")
    predicted_file = open(i7,"w")
    metric_file = open(i8,"w")


    (index_to_word,word_to_index) = formatData(index_to_word_raw,1)
    (index_to_tag,tag_to_index) = formatData(index_to_tag_raw,1)
    #hmmprior = formatData(hmmprior_raw,2) #np arr
    # num_tag_index x num_word_index
    #hmmemit = formatData(hmmemit_raw,3)
     # num_tag_index x num_tag_index
    #hmmtrans = formatData(hmmtrans_raw,3)
    hmmprior = np.array([1 for j in range(len(index_to_tag))])
    hmmtrans = np.array([np.array([1 for i in range(len(index_to_tag))])
                         for j in range(len(index_to_tag))])
    hmmemit = np.array([np.array([1 for i in range(len(index_to_word))])
                        for j in range(len(index_to_tag))])


    test_input = []
    for line in test_input_raw.readlines():
        tokens = line.strip().split(" ")
        token_list = []
        for t in tokens:
            token = t.split("_")
            word_idx = word_to_index[token[0]] #saving index of word
            tag_idx = tag_to_index[token[1]] #saving index of tag
            token_list.append((word_idx,tag_idx))
        test_input.append(token_list)

    #########################################################################
    # Prediction
    #########################################################################

    predictions = []
    log_likelihood = []
    num_correct = 0
    likelihood = 0.0


    for sentence in test_input:

        alpha,log_likelihood_list = getAlpha(sentence,hmmprior,hmmtrans,hmmemit)
        beta = getBeta(sentence,hmmemit,hmmtrans,index_to_tag)
        for row in alpha:
            row /= np.sum(row)
        for row in beta:
            row /= np.sum(row)

        for t,pair in enumerate(sentence):
            tag_res = []
            tag_index = pair[1]
            a_t,b_t = alpha[t],beta[t]
            p_t = a_t * b_t

            #find argmax
            p_max = None
            max_index = None
            for i in range(len(p_t)):
                if (p_max == None or p_max < p_t[i]):
                    p_max = p_t[i]
                    max_index = i
            tag_res.append((pair[0],max_index))
            if (max_index == tag_index):
                num_correct += 1
        predictions.append(tag_res)

    #get average log_likelihood
    avg_log_likelihood = sum(log_likelihood_list ) / len(log_likelihood_list)
    accuracy = num_correct / total

    # write output
    for prediction in predictions:
        toPrint = ""
        for pair in prediction:
            toPrint += index_to_word[pair[0]] + "_" + index_to_tag[tag[1]] + " "
        predicted_file.write(toPrint + "\n")

    metric_file.write("Average Log-Likelihood: " + str(log_likelihood) +
                      "\nAccuracy: " + str(accuracy))

    test_input_raw.close()
    index_to_word_raw.close()
    index_to_tag_raw.close()
    hmmprior_out.close()
    hmmemit_out.close()
    hmmtrans_out.close()
    predicted_file.close()
    metric_file.close()
