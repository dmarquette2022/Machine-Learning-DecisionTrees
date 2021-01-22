import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]



    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """



    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    i=0

    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    truePos = 0
    for truth in actual:
        if(truth == predictions[i]):
            if(truth):
                truePos = truePos + 1
            else:
                trueNeg = trueNeg + 1
        else:
            if(predictions[i]):
                falsePos = falsePos+1
            else:
                falseNeg = falseNeg+1
        i=i+1

    confMatrix = np.array([[trueNeg, falsePos],[falseNeg, truePos]])
    return confMatrix





def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confMatrix = confusion_matrix(actual,predictions)
    top = confMatrix[0][0] + confMatrix[1][1]
    bottom = top + confMatrix[0][1] + confMatrix[1][0]
    acc = (float)(top/bottom)
    return acc

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confMatrix = confusion_matrix(actual, predictions)
    trueNeg = confMatrix[0][0]
    truePos = confMatrix[1][1]
    falsePos = confMatrix[0][1]
    falseNeg = confMatrix[1][0]

    recall = truePos/(truePos + falseNeg)
    precision = truePos/(truePos + falsePos)

    return precision, recall


def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = precision_and_recall(actual, predictions)

    top = precision * recall
    bottom = precision+recall
    f1 = 2 * (top/bottom)

    return f1



