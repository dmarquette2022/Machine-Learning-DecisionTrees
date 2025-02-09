import numpy as np

class PriorProbability():
    def __init__(self):
        """
        This is a simple classifier that only uses prior probability to classify 
        points. It just looks at the classes for each data point and always predicts
        the most common class.

        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N 
                examples.
        Output:
            VOID: You should be updating self.most_common_class with the most common class
            found from the prior probability.
        """

        trueCnt = 0
        falseCnt = 0

        for eval in targets:
            if eval == 1.0:
                trueCnt += 1
            else:
                falseCnt = falseCnt + 1
        
        if trueCnt >= falseCnt:
            self.most_common_class = float(1)
        else:
            self.most_common_class = float(0)

    def predict(self, data):
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
        predictions = np.zeros(data.shape[0])
        if self.most_common_class == 1:
            predictions = np.ones(data.shape[0])
        return predictions