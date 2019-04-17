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





def forwardbackward(words, word2index, index2word, tag2index, index2tag, prior, trans, emit):
	alpha = np.zeros((len(words), len(tag2index.keys())))
	for j in range(0, len(tag2index.keys())):
		alpha[0][j] = prior[j] * emit[j][word2index[words[0]]]
	if alpha.shape[0] > 1:
		alpha[0] /= np.sum(alpha[0])
	for t in range(1, len(words)):
		for j in range(0, len(tag2index.keys())):
			tot = 0.0
			for k in range(0, len(tag2index.keys())):
				tot += alpha[t - 1][k] * trans[k][j]
			alpha[t][j] = emit[j][word2index[words[t]]] * tot
		if t != len(words) - 1:
			alpha[t] /= np.sum(alpha[t])
	log_likelihood = np.log(np.sum(alpha[-1]))

	beta = np.zeros((len(words), len(tag2index.keys())))
	prob = alpha * beta
	tags_index = np.argmax(prob, axis = 1)
	return [index2tag[index] for index in tags_index], log_likelihood

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
    num_word = len(index_to_word.keys())
    num_tag = len(index_to_tag.keys())
    hmmprior = formatData(hmmprior_raw,2) #np arr
    # num_tag_index x num_word_index
    hmmemit = formatData(hmmemit_raw,3)
     # num_tag_index x num_tag_index
    hmmtrans = formatData(hmmtrans_raw,3)

    test_input = []
    for line in test_input_raw.readlines():
        tokens = line.strip("\n").split(" ")
        test_input.append(tokens)

    #Run
    predictions = []
    cumul_likelihood = 0.0
    total_tags = 0
    correct = 0
    count = 0

    for sentence in test_input:
        words,labels = [],[]
        prediction_list = []
        for pair in sentence:
            words.append(pair.split("_")[0])
            labels.append(pair.split("_")[1])

        # GET ALPHA
        alpha = np.zeros((len(words),num_tag))
        for j in range(0,num_tag):
            alpha[0][j] = hmmprior[j] * hmmemit[j][word_to_index[words[0]]]

        if (alpha.shape[0] > 1):
            alpha[0] /= np.sum(alpha[0])
        for t in range(1,len(words)):
            for i in range(0,num_tag):
                tmp = 0.0
                for k in range(0,num_tag):
                    tmp += alpha[t-1][k] * hmmtrans[k][i]
                alpha[t][i] = hmmemit[i][word_to_index[words[t]]] * tmp
            if (t < len(words)-1):
                alpha[t] /= np.sum(alpha[t])

        log_likelihood = np.log(np.sum(alpha[-1]))

        # GET BETA
    	beta = np.zeros((len(words), num_tag))

        tag_list = []
        for t in range(len(sentence)):
            count += 1
            curr_tag = labels[t]
            p = alpha[t] * beta[t]

            max_prob = None
            max_i = None
            for i in range(len(p)):
                if (max_i == None or p[i] > max_prob) :
                    max_prob = p[i]
                    max_i = i

            if (max_i == curr_tag):
                correct += 1
            tag_list.append((words[t], max_i))
        predictions.append(tag_list)
        cumul_likelihood += log_likelihood
        total_tags += len(tag_list)

        for i in range(len(tag_list)):
            if (index_to_tag[tag_list[i][1]] == labels[i]):
                correct += 1

    avg_log_likelihood = cumul_likelihood / float(len(test_input))
    accuracy = float(correct) / float(total_tags)

    metric_file.write("Average Log-Likelihood: " + str(avg_log_likelihood) + "\n")
    metric_file.write("Accuracy: " + str(accuracy))

    for prediction in predictions:
        toPrint = ""
        for t in prediction:
            toPrint += t[0] + "_" + index_to_tag[t[1]] + " "
        predicted_file.write(toPrint + "\n")

    test_input_raw.close()
    index_to_word_raw.close()
    index_to_tag_raw.close()
    hmmprior_raw.close()
    hmmemit_raw.close()
    hmmtrans_raw.close()
    predicted_file.close()
    metric_file.close()
