import numpy as np
import matplotlib.pyplot as plt
import math

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    # count the number of samples in the dataset
    num_samples = len(data)
    # count the number of samples in each class
    class_counts = {}
    for row in data:
        label = row[-1]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    # calculate the gini impurity
    gini = 1.0
    for label in class_counts:
        pi = class_counts[label] / num_samples
        gini -= pi ** 2
    


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # count the number of samples in the dataset
    num_samples = len(data)
    # count the number of samples in each class
    class_counts = {}
    for row in data:
        label = row[-1]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    # calculate the entropy
    entropy = 0.0
    for label in class_counts:
        pi = class_counts[label] / num_samples
        entropy -= pi * math.log2(pi)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """


    goodness = 0
    groups = {} # groups[feature_value] = data_subset
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    original_impurity = impurity_func(data[:, -1])
    
    # Split the data according to the feature and create subsets
    for feature_value in np.unique(data[:, feature]):
        groups[feature_value] = data[data[:, feature] == feature_value]
    
    # Calculate the impurity of each subset
    subset_impurities = np.array([impurity_func(groups[key][:, -1]) for key in groups])
    
    # Calculate the goodness of split
    subset_sizes = np.array([len(groups[key]) for key in groups])
    goodness = original_impurity- np.sum(subset_sizes / np.sum(subset_sizes) * subset_impurities) 
    
    if gain_ratio:
        # Calculate the split information for each subset
        split_info = -np.sum((subset_sizes / np.sum(subset_sizes)) * np.log2(subset_sizes / np.sum(subset_sizes)))
        
        if split_info != 0:
            # Calculate the gain ratio
            goodness = goodness / split_info
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness, groups
    


class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        class_counts={}
        for row in self.data:
          label = row[-1]
          if label not in class_counts:
            class_counts[label] = 0
          class_counts[label] += 1
        pred = max(class_counts, key=lambda k: class_counts[k])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)



    def ChiTest(self):
       if self.chi==1:
        return False
       degreeOfFreedom=len(self.children)-1
       chi_value=chi_table[degreeOfFreedom][self.chi]
       chi_sq=0
       Count=0
       class_counts={}
       for row in self.data:
          label = row[-1]
          if label not in class_counts:
            class_counts[label] = 0
          class_counts[label] += 1
       prob_data={label:class_counts[label]/len(self.data) for label in class_counts.keys()}
       for kid_node in self.children:
        NumberOfInstancesOfNode=len(kid_node.data)
        for key in class_counts.keys():
          Count = np.sum(kid_node.data[:, -1] == key)
          chi_sq+=(Count-prob_data[key]*NumberOfInstancesOfNode)**2/(NumberOfInstancesOfNode*prob_data[key])
       return (chi_sq<chi_value)
     
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.depth<self.max_depth and impurity_func(self.data)!=0:
         feature_goodness={}
         for i in range(self.data.shape[1]-1):
           feature_goodness[i]= goodness_of_split(self.data, i, impurity_func, self.gain_ratio)
         self.feature = max(feature_goodness, key=lambda k: feature_goodness[k][0])
         children_dict=feature_goodness[self.feature][1]
         for key in children_dict.keys():
            child=DecisionNode(children_dict[key], -1,self.depth+1,self.chi,self.max_depth,self.gain_ratio)
            self.add_child(child, key)
        else:
           self.children=[]
           self.children_values=[]
           self.terminal=True
           self.feature=-1
        if len(self.children)<2 or self.ChiTest() : 
            self.children=[]
            self.children_values=[]
            self.terminal=True
            self.feature=-1
           
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    

def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    nodes=[]
    root= DecisionNode(data, -1,0,chi,max_depth,gain_ratio)
    nodes.append(root)
    while len(nodes)!= 0:
       nodes[0].split(impurity)
       nodes=nodes+nodes[0].children
       nodes.remove(nodes[0])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    node=root
    while not node.terminal:
        instance_feature=instance[node.feature]
        if not instance_feature in node.children_values:
            break
        node=node.children[node.children_values.index(instance_feature)] 
    pred=node.pred
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    num_correct = 0
    for instance in dataset:
        prediction = predict(node, instance)
        if prediction == instance[-1]:
            num_correct += 1
    accuracy = num_correct / len(dataset)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        root=build_tree(data=X_train, impurity=calc_entropy,gain_ratio=True, max_depth=depth)
        training.append(calc_accuracy(root, X_train))
        testing.append(calc_accuracy(root, X_test))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return training, testing


def TreeTraverse(root):
    nodes=[]
    NumberOfNudes=0
    depth=0
    nodes.append(root)
    while len(nodes)!= 0:
        NumberOfNudes+=1
        if(nodes[0].depth>depth):
            depth=nodes[0].depth
        nodes=nodes+nodes[0].children
        nodes.remove(nodes[0])
    return NumberOfNudes,depth


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for chi_val in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        root=build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True,chi=chi_val)
        chi_training_acc.append(calc_accuracy(root, X_train))
        chi_testing_acc.append(calc_accuracy(root, X_test))
        depth.append(TreeTraverse(root)[1])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    n_nodes=TreeTraverse(node)[0]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






