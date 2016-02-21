import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores_max = np.max(scores)
    scores -= scores_max
    exp_scores = np.exp(scores)
    sums = np.sum(exp_scores)
    log_sums = np.log(sums)
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
  loss -=np.max(loss)
  loss = np.exp(loss)/np.sum(exp(loss))
        
  dW += 0.5 * reg * 2 * W
  dW /= num_train
  
   #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
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
  margin -=np.max(margin)
  margin = np.exp(margin)/np.sum(exp(margin)) 
  loss = np.sum(margin)/X.shape[0]
  loss += 0.5*reg*np.sum(W*W)
  loss -=np.max(loss)
  loss = np.exp(loss)/np.sum(exp(loss)) 
  dW += 0.5 * reg * 2 * W
  dW /= X.shape[0]
    

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

