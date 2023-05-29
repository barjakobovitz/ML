import numpy as np

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    """
    def HX(self, X):
        HxDenominator=1+np.power(np.e,-1*np.dot(self.theta.T,X))
        Hx=1/HxDenominator
        return Hx
    
    def GX(self,X,y):
        value=-y*np.log(self.HX(X))-(1-y)*np.log(1-self.HX(X))
        return value/len(X)


    def fit(self, X, y):
        
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
       
        
        # set random seed
        np.random.seed(self.random_state)
        ones_cols = np.ones((X.shape[0]))
        result = np.column_stack((X, ones_cols)).T
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.theta= np.zeros(len(X.T)+1) #np.random.random(size=len(result))
        
        for i in range(self.n_iter):
         loss = self.GX(result, y)
         gradient = np.dot(result, loss)
         self.theta = self.theta - self.eta * gradient
         self.thetas.append(self.theta.copy())
         self.Js.append(np.sum(self.GX(result, y),axis=0))
         if i > 0 and (self.Js[i-1]-self.Js[i-1]) < self.eps:
            break
      """ 
    def sigmoid(self, X):
      

      z = np.dot(X, self.theta)
      
      e_to_power = np.exp(-z)
      
      return 1 / (1 + e_to_power)
    
    def cost_function(self, X, y):
      
      cost0 = np.dot(y, np.log(self.sigmoid(X)))
      cost1 = np.dot((1-y), np.log(1-self.sigmoid(X)))
      cost = -((cost1 + cost0))/len(y) 
      return cost
    
    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)
        m = X.shape[0]
        self.theta = np.zeros((X.shape[1]) + 1)
        new_data = np.c_[np.ones((X.shape[0],1)),X]
        for i in range(self.n_iter):
          self.theta = self.theta - self.eta * np.dot(new_data.T,self.sigmoid(new_data) - y)
          self.thetas.append(self.theta)
          self.Js.append(self.cost_function(new_data,y))
          if i > 0 and (self.Js[-2] - self.Js[-1] < self.eps): # Checking if the loss value is less than (1e-8), if true -> break, else continue.
            break
    

         
            
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


    
    
    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ones_cols = np.ones((X.shape[0]))
        result = np.column_stack((X, ones_cols))
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        preds=np.dot(result,self.theta)
        for i in range (len(preds)):
          if preds[i]<=0.5:
            preds[i]=0
          else:
            preds[i]=1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None
    
    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
        # Shuffle the data
    cv_accuracy = None

    # Set random seed
    np.random.seed(random_state)

    # Shuffle the indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Split indices into folds
    fold_indices = np.array_split(indices, folds)

    accuracies = []

    # Perform cross validation
    for i in range(folds):
        # Get the training indices by concatenating all other fold indices
        train_indices = np.concatenate([fold_indices[j] for j in range(folds) if j != i])

        # Get the validation indices for the current fold
        val_indices = fold_indices[i]

        # Split data into train and validation sets
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Train the model on the training set
        algo.fit(X_train, y_train)

        # Predict the validation set
        y_pred = algo.predict(X_val)

        # Calculate accuracy on the validation set
        accuracy = np.mean(y_pred == y_val)
        accuracies.append(accuracy)

    # Calculate the cross-validation accuracy as the mean of individual fold accuracies
    cv_accuracy = np.mean(accuracies)

   

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy
    
    

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((data - mu) / sigma) ** 2
    p = coefficient * np.exp(exponent)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        numberOfSamples= data.shape[0]
        numberOfFeatures = data.shape[1]

        self.responsibilities = np.ones((numberOfSamples, self.k)) / self.k
        self.weights = np.ones(self.k) / self.k
        random_indices = np.random.choice(numberOfSamples, size=self.k, replace=False)
        self.mus = data[random_indices, :]
        self.sigmas = np.zeros((self.k, numberOfFeatures, numberOfFeatures))
        for i in range(self.k):
         self.sigmas[i] = np.eye(numberOfFeatures)
        self.costs = []
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        numberOfSamples= data.shape[0]
        responsibilities = np.zeros((numberOfSamples, self.k))

        for g in range(self.k):
          mu=self.mus[g]
          sigma=self.sigmas[g]
          probabilty=norm_pdf(data,mu,sigma)
          probabilty = np.squeeze(probabilty)
          weight=self.weights[g]
          responsibilities[:, g] = weight*probabilty

        sum_responsibilities = np.sum(responsibilities, axis=1, keepdims=True)
        responsibilities /= sum_responsibilities
        self.responsibilities=responsibilities
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        numberOfSamples = data.shape[0]
        numberOfFeatures = data.shape[1]

        # Update the weights
        sum_responsibilities = np.sum(self.responsibilities, axis=0)
        self.weights = sum_responsibilities / numberOfSamples

        # Update the means
        self.mus = np.dot(self.responsibilities.T, data) / sum_responsibilities[:, np.newaxis]

        # Update the covariances
        self.sigmas = np.zeros((self.k, numberOfFeatures, numberOfFeatures))
        for j in range(self.k):
          diff = data - self.mus[j]
          weighted_diff = self.responsibilities[:, j][:, np.newaxis] * diff
          self.sigmas[j] = np.dot(weighted_diff.T, diff) / sum_responsibilities[j]

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    def costFunction(self, data):
       cost=0
       
       for i in range(data.shape[0]):
         costPerSample=0
         sample = data[i]
         for j in range(self.k):
            mean = self.mus[j]
            covariance = self.sigmas[j]
            weight = self.weights[j]

            # Calculate the probability of the sample belonging to cluster j
            gaussian_prob = norm_pdf(sample, mean, covariance)

            # Weight the probability by the cluster weight
            weighted_prob = weight * gaussian_prob
            
            costPerSample-=np.log(weighted_prob)

            # Accumulate the likelihood
         cost += np.log(costPerSample)

         # Take the logarithm of the accumulated likelihood and add it to the overall log-likelihood
          

       return cost


    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.init_params(data)
        for i in range(self.n_iter):
          self.expectation(data)
          self.maximization(data)
          self.costs.append(self.costFunction(data))
          if i > 0 and (self.costs[-2] - self.costs[-1] < self.eps): 
            break  
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pdf = np.zeros_like(data)
    
    for weight, mu, sigma in zip(weights, mus, sigmas):
        gaussian_pdf = weight * (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-0.5 * ((data - mu)**2) / (sigma**2))
        pdf += gaussian_pdf
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }