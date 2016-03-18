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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
    scores=X[i].dot(W)  
    scores-=np.max(scores)
    lyi=-scores[y[i]]
    li=0
    for j in xrange(num_classes):     
      li+=np.exp(scores[j])
      lossi=lyi+np.log(li)
    for j in xrange(num_classes):
      dW[:,j]+=(np.exp(scores[j])/li-(j==y[i]))*X[i]      
    loss+=lossi
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /=num_train 
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW+= reg*W   
                  
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
  num_train=X.shape[0]
  num_classes=W.shape[1]
  scores=X.dot(W)  #N,C
  scores-=np.max(scores)
  loss=-np.sum(scores[np.arange(num_train),y])+np.sum(np.log(np.sum(np.exp(scores),axis=1))) 
  f=np.exp(scores.T)/np.sum(np.exp(scores),axis=1)
  fyi=np.zeros_like(f)
  fyi[y,np.arange(num_train)]=1
  dW=np.dot(X.T,(f-fyi).T)
  loss /= num_train 
  dW /=num_train 
  loss += 0.5 * reg * np.sum(W * W)
  dW+= reg*W  
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

