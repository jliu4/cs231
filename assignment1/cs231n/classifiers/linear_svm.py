import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
 
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    ddW=np.zeros(dW.shape)
    ddWyi=np.zeros(dW.shape[0])
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        ddW[:,j]=X[i] 
        ddWyi += ddW[:,j]
    ddW[:,y[i]]=-ddWyi  
    dW += ddW
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += 0.5 * reg * 2 * W
  dW /= num_train
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores=X.dot(W) #N,C
  correct_class_score=[scores[i,y[i]] for i in xrange(scores.shape[0])] # C
  margin = scores.T - correct_class_score + 1 # C,N
  factor = np.array((margin>0).astype(int)) #C,N
  factor_yi=-np.sum(factor,axis=0)  #N
  for i in xrange(scores.shape[0]): #N
    margin[y[i],i] = 0
    factor[y[i],i] = factor_yi[i]+1 #take out the yi=j

  dW=factor.dot(X).T
  margin = np.maximum(0,margin) 
  loss = np.sum(margin)/X.shape[0]
  loss += 0.5*reg*np.sum(W*W)
  dW += 0.5 * reg * 2 * W
  dW /= X.shape[0]
    
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
