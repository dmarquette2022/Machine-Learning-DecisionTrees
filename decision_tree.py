import numpy as np

class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)
        
        """
        self.attribute_names = attribute_names
        self.tree = None
        self.FirstTime = True

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        Output:
            VOID: It should update self.tree with a built decision tree.
        """
        self._check_input(features)
        self.tree = self.ID3(features, targets, self.attribute_names, None)
        print("done!")

    def mostCommonClass(targets):
        numPos = 0
        numNeg = 0
        for target in targets:
            if(target):
                numPos = numPos + 1
            else:
                numNeg = numNeg + 1

        if(numPos>=numNeg):
            return 1
        else:
            return 0

    
    def ID3(self, features, targets, attributes, val):
        t = Node()
        if(self.FirstTime):
            self.tree = t
            self.FirstTime = False
        numPos = 0
        numNeg = 0
        for fin in targets:
            if(fin):
                numPos=numPos +1
            else:
                numNeg = numNeg + 1
        if numPos == 0:
            t.value = 0
            return t
        elif numNeg == 0:
            t.value = 1
            return t
        
        if(len(attributes) == 0):
            if(numPos>= numNeg):
                t.value = 1
            else:
                t.value = 0
            return t
        
        maxGain = None
        A = 0
        maxIndex = 0
        for index, att in enumerate(attributes):
            currGain = information_gain(features, index, targets)
            if maxGain == None:
                maxIndex = index
                maxGain = currGain
            elif currGain > maxGain:
                maxIndex = index
                maxGain = currGain

        newAttributes = [i for i in attributes if i is not attributes[maxIndex]]
        maxOriginalAttIndex = self.attribute_names.index(attributes[maxIndex])

        s0_features = features[features[:, maxOriginalAttIndex] == 0]
        s0_target = targets[features[:, maxOriginalAttIndex] == 0]
        s1_features = features[features[:, maxOriginalAttIndex] == 1]
        s1_target = targets[features[:, maxOriginalAttIndex] == 1]



        return Node(val, attributes[maxIndex], maxOriginalAttIndex, [
            self.ID3(s0_features, s0_target, newAttributes, 0),
            self.ID3(s1_features, s1_target, newAttributes, 1)
        ])
        """

        options = [0,1]
        column = features[:,A]
        D_a = np.ones((1, features.shape[1]))
        firstTime = True
        target_a = []
        print(A)
        t.attribute_index = A
        t.attribute_name = self.attribute_names[A]
        for option in options:
            i = 0
            for row in column:
                if row == option:
                    if(firstTime):
                        D_a = features[i,:]
                        firstTime = False
                    else:
                        D_a = np.vstack((D_a, features[i,:]))
                    target_a.append(targets[i])
                i=i+1
            if(len(D_a) == 0):
                tPrime = Node()
                tPrime.value = self.mostCommonClass(targets)
                t.branches.append(tPrime)
            else:
                t.branches.append(self.ID3(D_a, target_a, attributes))
        return t
        """




    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predicitons 
            for the input data.
        """
        self._check_input(features)

        predictions = np.ones(features.shape[0])
        i=0
        for row in features:
            curr = self.tree
            while len(curr.branches) != 0:
                propAtt = curr.attribute_index
                if row[propAtt]:
                    curr = curr.branches[1]
                else:
                    curr = curr.branches[0]
            predictions[i] = curr.value
            i=i+1

        return predictions
        







    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """

    sCntZero = 0
    sCntOnes = 0
    sCntTotal = 0
    
    """
    print("TARGETS")
    print(targets)

    print("Features")
    print(features)

    print(len(targets))
    print("mokey")
    print(len(features))

    print(attribute_index)
    """
    

    for item in targets:
        if item == 0:
            sCntZero = sCntZero + 1
        else:
            sCntOnes = sCntOnes + 1
        sCntTotal = sCntTotal + 1
    
    probZero = sCntZero/sCntTotal
    probOne = sCntOnes/sCntTotal

    sEntrop = (-probZero * np.log2(probZero)) + (-probOne * np.log2(probOne))

    zeroCntOne = 0
    oneCntOne = 0

    zeroCntZero = 0
    oneCntZero = 0

    i = 0
    #[attribute]Cnt[Target]
    for row in features:
        test = row[attribute_index]
        if(test):
            if targets[i]:
                oneCntOne = oneCntOne + 1
            else:
                oneCntZero = oneCntZero + 1
        else:
            if targets[i]:
                zeroCntOne = zeroCntOne + 1
            else:
                zeroCntZero = zeroCntZero + 1
        i = i+1

    total = oneCntOne + oneCntZero + zeroCntZero + zeroCntOne
    attOne = oneCntOne + oneCntZero
    attZero = zeroCntOne + zeroCntZero
    if oneCntOne == 0 or oneCntZero == 0:
        EntropyAttOne = 0
    else:
        probOneAttOne = oneCntOne/attOne
        probZeroAttOne = oneCntZero/attOne
        EntropyAttOne = (-probOneAttOne * np.log2(probOneAttOne)) + (-probZeroAttOne * np.log2(probZeroAttOne))
    if zeroCntOne == 0 or zeroCntZero == 0:
        EntropyAttZero = 0
    else:
        probOneAttZero = zeroCntOne/attZero
        probZeroAttZero = zeroCntZero/attZero
        EntropyAttZero = (-probOneAttZero * np.log2(probOneAttZero)) + (-probZeroAttZero * np.log2(probZeroAttZero))



    totEntropy = (attZero/total)*(EntropyAttZero) + (attOne/total)*(EntropyAttOne)

    return sEntrop - totEntropy
    


    




if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
