from __future__ import division
import sys

#########################################################################
# model_1_format
# takes in raw data and formats to bag-of-word representation
# param : dict, data, result file path
# returns : None, directly writes to given output file
#########################################################################

def model_1_format(d,data,output):
    for line in data.splitlines():
        appeared_words = set()
        labelSplit = line.split("\t")
        label,words = labelSplit[0],labelSplit[1]
        output.write("%s\t" % label)
        for word in words.split(" "):
            if ((word in d) and (word not in appeared_words)):
                output.write("%s:1\t" % d[word])
                appeared_words.add(word)
        output.write("\n")
    return None

#########################################################################
# model_2_format
# takes in raw data and formats to trimmed bag-of-word representation
# param : dict, data, result file path
# returns : None, directly writes to given output file
#########################################################################

def model_2_format(d,data,output):
    t = 4 # Given threshold by the assignment
    data_words = dict()
    for line in data.splitlines():
        appeared_words = set()
        labelSplit = line.split("\t")
        label,words = labelSplit[0],labelSplit[1]
        output.write("%s\t" % label)
        words_list = words.split(" ")
        for word in words_list:
            if word not in data_words: data_words[word] = 1
            else: data_words[word] += 1
        for word in words_list:
            if (data_words[word] < t):
                if ((word in d) and (word not in appeared_words)):
                    output.write("%s:1\t" % d[word])
                    appeared_words.add(word)
        output.write("\n")
    return None

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
    formatted_train_out = open(i5,"w")
    formatted_validation_out = open(i6,"w")
    formatted_test_out = open(i7,"w")
    feature_flag = int(i8)

#########################################################################
#   Format given inputs
#########################################################################
    dict_raw = dict_input.read()
    word_dict = format_dict(dict_raw)

    train_data = train_input.read()
    validation_data = validation_input.read()
    test_data = test_input.read()

    # Use model 1
    if (feature_flag == 1):
        model_1_format(word_dict,train_data,formatted_train_out)
        model_1_format(word_dict,validation_data,formatted_validation_out)
        model_1_format(word_dict,test_data,formatted_test_out)
    # Use model 2
    elif (feature_flag == 2):
        model_2_format(word_dict,train_data,formatted_train_out)
        model_2_format(word_dict,validation_data,formatted_validation_out)
        model_2_format(word_dict,test_data,formatted_test_out)
    else : raise Exception("feature_flag should be 1 or 2")

    train_input.close()
    validation_input.close()
    test_input.close()
    dict_input.close()
    formatted_train_out.close()
    formatted_validation_out.close()
    formatted_test_out.close()
