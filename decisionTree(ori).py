import sys
import csv
from collections import Counter
import math
from collections import defaultdict


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
class Node:
    def __init__(self,key = None):
        self.left = None
        self.right = None
        self.val = key #attribute
        self.data = None
        self.parent = None
        self.label = None
        self.depth = None

#helper for opening a file and store data as dictionary
def store_data_dict(input_file):
    f  = open(input_file, 'r')
    input_f = csv.reader(f, delimiter='\t')
    d = dict()
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
        tmp += ((S_dict.get(key))/total ) * (math.log2((S_dict.get(key))/total))
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
def train_stump(root, target_attr, remain_attr, max_depth, depth):
    guess = majority_vote_label(root.val.get(target_attr)) #most frequent label
    root.label = guess
    if max_depth == 0: #depth = 0, root
        return root
    elif len(remain_attr) == 0: #attrbs empty
        return root
    elif root.depth == int(max_depth): #don't grow larger than max-depth
        return root
    else:
        #choose the best attribute to split
        targ_attr_vals = root.val.get(target_attr)
        info_gains = []
        for attr in remain_attr:
            attr_vals = root.val.get(attr)
            mutual_info = get_mutual_info(attr_vals, targ_attr_vals)
            info_gains.append(mutual_info)
        #only split if mutual info > 0
        if max(info_gains) <= 0:
            return root
        best_attr = remain_attr[info_gains.index(max(info_gains))]
        best_attr_vals = root.val.get(best_attr)
        unique_best_attr_vals = list(set(best_attr_vals))
        val_no = unique_best_attr_vals[0]
        val_yes = unique_best_attr_vals[1]
        subset_data_no = get_subset_data(root.val, best_attr_vals, val_no)
        subset_data_yes = get_subset_data(root.val, best_attr_vals, val_yes)
        #descendant
        left_node = Node(subset_data_no)
        left_node.label = best_attr
        right_node = Node(subset_data_yes)
        right_node.label = best_attr
        # recursively call
        copied = remain_attr[:]
        copied.remove(best_attr)
        left_node.depth = depth
        right_node.depth = depth
        left = train_stump(left_node, target_attr, copied, max_depth, depth + 1)
        left.parent = best_attr
        remain_attr.remove(best_attr)
        right = train_stump(right_node, target_attr, remain_attr, max_depth, depth + 1)
        right.parent = best_attr
        root.left = left
        root.right = right
        return root

def train(train_data, target_attr, remain_attr, max_depth):
    root = Node(train_data)
    root.depth = 0
    guess = majority_vote_label(root.val.get(target_attr)) #most frequent label
    root.label = guess
    if int(max_depth) == 0:
        return root
    learned_tree = train_stump(root, target_attr, remain_attr, max_depth, depth = 1)
    return learned_tree

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
def pretty_print(root, data, target_attr, target_labels, depth = 0):
    if root:
        if depth == 0:
            stat = get_stat(data.get(target_attr), target_labels)
            print(stat, end ="\n")
        else:
            stat =  get_stat(root.val.get(target_attr),target_labels)
            label = root.val.get(root.label)[0]
            print( "|" * root.depth, root.label, "=",  label, ":", stat,"\n", end = "")
        pretty_print(root.left, data, target_attr, target_labels, depth + 1)
        pretty_print(root.right,data, target_attr, target_labels, depth + 1)


def get_error(probs):
    if len(set(probs)) == 1: #classified
        return 0
    min_num = min(probs)
    min_count = probs.count(min_num)
    return min_count

def get_training_error(root, data, targ_val):
    if root:
        if root.left == None and root.right == None:
            targ_vals = root.val.get(targ_val)
            error = get_error(targ_vals)
            return error
        return get_training_error(root.left, data, targ_val) + \
               get_training_error(root.right, data, targ_val)

#predict
def predict_labels(root, row, target_attr):  # test
    if root:
        if root.left == None and root.right == None: #leaf
            return majority_vote_label(root.val.get(target_attr))
        else:#recurse
            #find matching value
            attr = root.left.label
            if attr in row.keys():
                val = row.get(attr)
                if val in root.left.val.get(attr):
                    return predict_labels(root.left, row, target_attr)
                else:
                    return predict_labels(root.right, row, target_attr)
            return majority_vote_label(root.val.get(target_attr))


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

def write_file(txt, out):
    txt += "\n"
    out.write(txt)

def test(root, rows, targ_attr, output_file):
    out = open(output_file, 'w')
    total = len(rows)
    tally = 0
    for row in rows:
        predicted_label = predict_labels(root, row, targ_attr)
        # print(predicted_label)
        write_file(predicted_label, out)
        actual_label = row.get(targ_attr)
        if predicted_label != actual_label:
            tally += 1
    out.close()
    return tally / total

if __name__ == '__main__':
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    max_depth = sys.argv[3]
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

    #test attrs
    test_attr = list(test_data.keys())
    test_target_attr = test_attr[-1]
    test_remain_attr = test_attr[:len(attr) - 1]  # names of attributes
    test_target_labels = list(set(test_data.get(test_target_attr)))

    #train tree
    learned_tree = train(train_data, target_attr, remain_attr, max_depth)

    #test tree
    train_rows = get_rows(train_data, target_attr)
    test_rows = get_rows(test_data, test_target_attr)

    test(learned_tree, train_rows, target_attr ,train_out)
    #test error
    tests_error = test(learned_tree, test_rows, test_target_attr, test_out)
    #train_error
    train_error = get_training_error(learned_tree, train_data,target_attr)

    # metrics
    metrics_out_file = open(metrics_out, 'w')
    train_total_error = train_error / len(train_data.get(target_attr))
    txts = "error (train) : " + str(train_total_error) + "\n" + "error (test) : " + str(tests_error)

    metrics_out_file.write(txts)
    metrics_out_file.close()

    # print
    pretty_print(learned_tree, train_data, target_attr, target_labels, depth=0)
    # pretty_print(learned_tree, test_data, test_target_attr, test_target_labels, depth=0)







