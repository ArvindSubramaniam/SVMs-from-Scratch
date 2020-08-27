import numpy as np

train_data=np.load("train_data.txt",allow_pickle=True)

def train_svm(train_data, train_label, C):
  """Train linear SVM (primal form)

  Argument:
    train_data: N*D matrix, each row as a sample and each column as a feature
    train_label: N*1 vector, each row as a label
    C: tradeoff parameter (on slack variable side)

  Return:
    w: feature vector (column vector)
    b: bias term
  """


def test_svm(test_data, test_label, w, b):
  """Test linear SVM

  Argument:
    test_data: M*D matrix, each row as a sample and each column as a feature
    test_label: M*1 vector, each row as a label
    w: feature vector
    b: bias term

  Return:
    test_accuracy: a float between [0, 1] representing the test accuracy
  """
