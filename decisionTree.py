import sys
import csv
from collections import Counter
import math
from collections import defaultdict
from collections import OrderedDict
import matplotlib.pyplot as plt


#Goal:
#learn a decision tree with a specified maximum depth,
#print the decision tree in a specified format
#predict that labels of the training and testing examples
#calculate training and testing errors

#Rules:
#only split on attribute if the mutual information is > 0
#do not grow beyond the max-depth specified by the command line
#use the majority vote of the labels at each leaf to make classification


# store attributes, constructor
# represents an individual node

# store attributes, constructor
# represents an individual node
class Node:
    def __init__(self,key = None, left = None, right= None):
        self.left = left
        self.right = right
        self.val = key #attribute
        self.depth = None
        self.attr = None

#helper for opening a file and store data as dictionary
def store_data_dict(input_file):
    f  = open(input_file, 'r')
    input_f = csv.reader(f, delimiter='\t')
    d = OrderedDict()
    counter = 0
    for row in input_f:
        for i in range(len(row)):
            if counter == 0:
                attrbs = row
                d[row[i]] = []
            else:
                d[attrbs[i]].append(row[i])
        counter += 1
    return d

def calculate_entropy(S):
    total = len(S)
    S_dict= defaultdict(int)
    for val in S:
        S_dict[val] += 1
    tmp = 0
    for key in S_dict:
        tmp += ((S_dict.get(key))/total )* (math.log2((S_dict.get(key))/total))
    return -tmp

def get_mutual_info(attr_vals, targ_attr_vals):
    marginal = calculate_entropy(targ_attr_vals)
    if len(set(attr_vals)) == 1:
        cond_entropy = calculate_entropy(targ_attr_vals)
    else:
        indices_a = [i for i, val in enumerate(attr_vals) if val == list(set(attr_vals))[0]]
        targ_vals_a = [targ_attr_vals[i]  for i in indices_a]
        cond_entropy_a = calculate_entropy(targ_vals_a)

        indices_b = [i for i, val in enumerate(attr_vals) if val == list(set(attr_vals))[1]]
        targ_vals_b = [targ_attr_vals[i] for i in indices_b]
        cond_entropy_b = calculate_entropy(targ_vals_b)
        total = len(targ_attr_vals)
        cond_entropy = (( attr_vals.count(list(set(attr_vals))[0]) /total )* cond_entropy_a)  + \
                       ((attr_vals.count(list(set(attr_vals))[1]) /total ) * cond_entropy_b)
    return marginal - cond_entropy

def get_subset_data(data, best_attr_vals, specific_attr_val):
    specific_attr_val_index = [i for i, val in enumerate(best_attr_vals) if val == specific_attr_val]
    subset_data = dict()
    for key in list(data.keys()):
        subset_data[key] = [data.get(key)[i] for i in specific_attr_val_index]
    return subset_data

def majority_vote_label(data_value):
    counter_dict = Counter(data_value)
    return counter_dict.most_common(1)[0][0]

#training stump(tree with only one level)
def train_stump(data, best_attr, target_attr, remain_attr, max_depth, depth):
    guess = majority_vote_label(data.get(target_attr))
    leaf = Node(data)
    leaf.depth = depth
    leaf.label = guess
    leaf.attr = best_attr
    if len(remain_attr) == 0: #attrbs empty
        return leaf
    elif leaf.depth == int(max_depth): #don't grow larger than max-depth
        return leaf
    else:
        #choose the best attribute to split
        targ_attr_vals = data.get(target_attr)
        info_gains = []
        for attr in remain_attr:
            attr_vals = data.get(attr)
            mutual_info = get_mutual_info(attr_vals, targ_attr_vals)
            info_gains.append(mutual_info)
        best_attr = remain_attr[info_gains.index(max(info_gains))]
        best_attr_vals = data.get(best_attr)
        #only split if mutual info > 0 and level < max depth
        if max(info_gains) <= 0 and leaf.depth < int(max_depth):
            return leaf
        if max(info_gains) > 0 and leaf.depth < int(max_depth):
            unique_best_attr_vals = list(set(best_attr_vals))
            val_no = unique_best_attr_vals[0]
            val_yes = unique_best_attr_vals[1]
            copied = remain_attr[:]
            copied.remove(best_attr)
            subset_data_no = get_subset_data(data, best_attr_vals, val_no)
            subset_data_yes = get_subset_data(data, best_attr_vals, val_yes)
            left = train_stump(subset_data_no, best_attr, target_attr, copied, max_depth, depth + 1)
            remain_attr.remove(best_attr)
            right = train_stump(subset_data_yes, best_attr , target_attr, remain_attr, max_depth, depth + 1)
            leaf.left = left
            leaf.right = right
            return leaf

def train(train_data , target_attr, remain_attr, max_depth):
    root = Node(train_data)
    root.depth = 0
    root.label = majority_vote_label(train_data.get(target_attr))
    if int(max_depth) == 0:
        return root
    if len(remain_attr) == 0:
        return root
    targ_attr_vals = train_data.get(target_attr)
    info_gains = []
    for attr in remain_attr:
        attr_vals = root.val.get(attr)
        mutual_info = get_mutual_info(attr_vals, targ_attr_vals)
        info_gains.append(mutual_info)
    best_attr = remain_attr[info_gains.index(max(info_gains))]
    best_attr_vals = root.val.get(best_attr)
    if max(info_gains) <= 0:
        return root
    elif max(info_gains) > 0 and root.depth < int(max_depth):
        unique_best_attr_vals = list(set(best_attr_vals))
        val_no = unique_best_attr_vals[0]
        val_yes = unique_best_attr_vals[1]
        copied = remain_attr[:]
        copied.remove(best_attr)
        subset_data_no = get_subset_data(train_data, best_attr_vals, val_no)
        subset_data_yes = get_subset_data(train_data, best_attr_vals, val_yes)
        left = train_stump(subset_data_no, best_attr ,target_attr, copied, max_depth, depth = 1)
        remain_attr.remove(best_attr)
        right = train_stump(subset_data_yes,best_attr , target_attr, remain_attr, max_depth, depth = 1)
        root.left = left
        root.right = right
        return root





def get_stat(target_data, target_labels):
    if len(target_labels) == 1:
        y = target_labels[0]
        y_count = target_data.count(y)
        stat = "[%d" %(y_count) + " %s"  %(y) + "]"
    else:
        y1 = target_labels[0]
        y2 = target_labels[1]
        y1_count = target_data.count(y1)
        y2_count = target_data.count(y2)
        stat = "[%d" % (y1_count) + " %s" %(y1) + " /%d" %(y2_count) + " %s" %(y2) + "]"
    return stat

#DFS order print
def pretty_print(root, data, target_attr, target_labels):
    if root:
        if root.depth == 0:
            stat = get_stat(root.val.get(target_attr), target_labels)
            print(stat, end ="\n")
        else:
            stat =  get_stat(root.val.get(target_attr),target_labels)
            label = root.val.get(root.attr)[0]
            print( "|" * root.depth, root.attr, "=",  label, ":", stat,"\n", end = "")
        pretty_print(root.left, data, target_attr, target_labels)
        pretty_print(root.right,data, target_attr, target_labels)


#predict
def predict_labels(root, row, target_attr):  # test
    if root:
        if (root.left == None or root.right == None):#leaf
            return root.label
        else:
            attr = root.left.attr
            if attr in row.keys():
                val = row.get(attr)
                if val in root.left.val.get(attr):
                    return predict_labels(root.left, row, target_attr)
                else:
                    return predict_labels(root.right, row, target_attr)


def get_rows(data, targ_attr):
    rows = []
    attrs = data.keys()
    total_len = len(data.get(targ_attr))
    for i in range (total_len):
        row = dict()
        for attr in attrs:
            row[attr] = data.get(attr)[i]
        rows.append(row)
    return rows



if __name__ == '__main__':
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    max_depth = sys.argv[3]
    # print(max_depth)
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    #store data
    train_data = store_data_dict(train_in)
    test_data = store_data_dict(test_in)

    #get attributes
    #train attrs
    attr = list(train_data.keys())
    target_attr = attr[-1]
    remain_attr = attr[:len(attr)-1] #names of attributes
    target_labels = list(set(train_data.get(target_attr)))
    print(len(remain_attr))

    depths = []
    for i in range(len(remain_attr) + 1):
        depths.append(i)

    train_errors = []
    test_errors = []

    # for max_depth in depths:
    test_attr = list(test_data.keys())
    test_target_attr = test_attr[-1]
    test_remain_attr = test_attr[:len(attr) - 1]  # names of attributes
    test_target_labels = list(set(test_data.get(test_target_attr)))

    #train tree
    learned_tree = train(train_data, target_attr, remain_attr, max_depth)

    #test tree
    train_rows = get_rows(train_data, target_attr)
    test_rows = get_rows(test_data, test_target_attr)


    #test train_tree
    train_labels_out = open(train_out, 'w')
    train_total = len(train_rows)
    train_tally = 0
    count = 0
    for row in train_rows:
        predicted_label = predict_labels(learned_tree, row, target_attr)
        txt = predicted_label + "\n"
        train_labels_out.write(txt)
        actual_label = row.get(target_attr)
        if predicted_label != actual_label:
            # print('here', count, predicted_label, actual_label)
            train_tally += 1
    train_error = train_tally / train_total
    train_error = round(train_error, 4)
    train_errors.append(train_error)

    #test test tree
    test_labels_out = open(test_out, 'w')
    test_total = len(test_rows)
    tally = 0
    for row in test_rows:
        predicted_label = predict_labels(learned_tree, row, test_target_attr)
        tst_txt = predicted_label + "\n"
        test_labels_out.write(tst_txt)
        actual_label = row.get(test_target_attr)
        if predicted_label != actual_label:
            # print('here', count, predicted_label, actual_label)
            tally += 1
    test_error = tally / test_total
    test_error = round(test_error, 4)
    test_errors.append(test_error)
    
    print(depths)
    print(train_errors)
    print(test_errors)
    fig = plt.figure()
    plt.errorbar(depths, train_errors, label='Train Error')
    plt.errorbar(depths, test_errors, label='Test Error')
    plt.legend(loc='lower right')
    plt.show()

 

    train_error = get_training_error(learned_tree, target_attr, len(train_rows))

    # metrics
    metrics_out_file = open(metrics_out, 'w')
    # train_total_error = train_error / len(train_data.get(target_attr))
    txts = "error (train) : " + str(train_error) + "\n" + "error (test) : " + str(test_error)
    print(txts)

    metrics_out_file.write(txts)
    metrics_out_file.close()
    train_labels_out.close()
    test_labels_out.close()

    pretty_print(learned_tree, train_data, target_attr, target_labels)




