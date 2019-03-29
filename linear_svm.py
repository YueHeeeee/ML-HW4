import numpy as np

##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, C):
  """
  SVM hinge loss function for two class problem

  Inputs:
  - theta: A numpy vector of size d containing coefficients.
  - X: A numpy array of shape mxd 
  - y: A numpy array of shape (m,) containing training labels; +1, -1
  - C: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to theta; an array of same shape as theta
"""

  m, d = X.shape
  grad = np.zeros(theta.shape)
  J = 0

  ############################################################################
  # TODO                                                                     #
  # Implement the binary SVM hinge loss function here                        #
  # 4 - 5 lines of vectorized code expected                                  #
  ############################################################################

  h_x = np.dot(X, theta)

  J = np.sum(theta ** 2)/(2*m) + C * np.mean( np.maximum(np.zeros(m),1 - y*h_x) ) 

  indx = ((y*h_x)<1).astype("float")

  grad = 1/m*theta + C * np.mean(indx[:,None]*(-y[:,None]*X), axis=0)

  # print(grad)



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, grad

##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension d, there are K classes, and we operate on minibatches
  of m examples.

  Inputs:
  - theta: A numpy array of shape d X K containing parameters.
  - X: A numpy array of shape m X d containing a minibatch of data.
  - y: A numpy array of shape (m,) containing training labels; y[i] = k means
    that X[i] has label k, where 0 <= k < K.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss J as single float
  - gradient with respect to weights theta; an array of same shape as theta
  """

  K = theta.shape[1] # number of classes
  m = X.shape[0]     # number of examples

  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Compute the loss function and store it in J.                              #
  # Do not forget the regularization term!                                    #
  # code above to compute the gradient.                                       #
  # 8-10 lines of code expected                                               #
  #############################################################################

  
  for j in range(K):
      for i in range(m):
          if j != y[i]:
              h_x = np.dot(theta[:,j],X[i]) -  np.dot(theta[:,y[i]],X[i]) + delta
              J += np.maximum(0,h_x)
              if h_x > 0:
                dtheta[:,j] += X[i]
                dtheta[:,y[i]] -= X[i]

  J = reg*np.sum(theta ** 2)/(2*m) + J/m
  dtheta = reg*theta/m + dtheta/m



  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dtheta.            #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return J, dtheta


def svm_loss_vectorized(theta, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in variable J.                                                     #
  # 8-10 lines of code                                                        #
  #############################################################################


  K = theta.shape[1] # number of classes
  m = X.shape[0]     # number of examples

  term1 = np.dot(X,theta)
  term2 = np.dot(X,theta)[range(m),y]
  #L = (term1 - term2[:,None]).reshape((-1,1))
  L = (term1 - term2[:,None])
  #print(L.shape)
  #print((L!=0).shape)
  L[L!=0] += delta
  J = np.maximum(0,L)
  J = reg*np.sum(theta ** 2)/(2*m) + np.sum(J)/m



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dtheta.                                       #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  indx = (L>0).astype(int)
  indx[range(m),y] = -np.sum(indx, axis=1) 
  dtheta = reg * theta/m + np.dot(X.T, indx)/m

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, dtheta
