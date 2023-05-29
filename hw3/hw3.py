import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.1,
            (1, 1): 0.6
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.3,
            (1, 1): 0.4
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.1,
            (0, 1): 0.2,
            (1, 0): 0.4,
            (1, 1): 0.3
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.04,
            (0, 0, 1): 0.04,
            (0, 1, 0): 0.16,
            (0, 1, 1): 0.06,
            (1, 0, 0): 0.06,
            (1, 0, 1): 0.16,
            (1, 1, 0): 0.24,
            (1, 1, 1): 0.24,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for key in X_Y.keys():
            if np.isclose(X_Y[key],X[key[0]]*Y[key[1]]):
                return False
        return True
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for key in X_Y_C.keys():
            X_given_C=X_C[(key[0],key[2])]/C[key[2]]
            Y_given_C=Y_C[(key[1],key[2])]/C[key[2]]
            XandY_given_C=X_Y_C[key]/C[key[2]]
            if np.isclose(X_given_C*Y_given_C,XandY_given_C):
                return True
        return False

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    p=(np.power(rate,k))*(np.power(np.e,-rate))/np.math.factorial(k)  
    log_p=np.log(p)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    #p(x|A)
    likelihoods = np.zeros(len(rates))
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for i in range(len(rates)):
        for sample in samples:
            likelihoods[i]+=poisson_log_pmf(sample, rates[i])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    max_likelihood=likelihoods[0]
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for i,likelihood in enumerate(likelihoods):
        if likelihood>max_likelihood:
            max_likelihood=likelihood
            rate=rates[i] 
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    mean=np.mean(samples)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    power_of_e = (((x - mean)**2) / (2*(std**2)))
    e_to_the_power = np.power(np.e,-power_of_e)
    sqrt = 1 / np.sqrt(2 * np.pi * (std**2))
    p = e_to_the_power * sqrt
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.mean=np.zeros(np.shape(dataset)[1]-1)
        self.std=np.zeros(np.shape(dataset)[1]-1)
        self.dataset=dataset
        self.class_value=class_value

        last_col = self.dataset[:, -1]  # Select the last column
        mask = last_col == self.class_value  # Create a boolean mask where the last column equals 1
        result = self.dataset[mask]  # Select the rows where the mask is True
        for i in range(len(self.mean)):
            self.mean[i]=np.mean(result[:,i],axis=0)
        for i in range(len(self.std)):
             self.std[i]=np.std(result[:,i],axis=0)

        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        last_col = self.dataset[:, -1]  # Select the last column
        mask = last_col == self.class_value  # Create a boolean mask where the last column equals 1
        result = self.dataset[mask]
        prior=len(result)/len(self.dataset)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for i in range(len(self.mean)):
            likelihood*=normal_pdf(x[i], self.mean[i], self.std[i])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        posterior=self.get_prior()*self.get_instance_likelihood(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0=ccd0
        self.ccd1=ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        if self.ccd1.get_instance_posterior(x)>self.ccd0.get_instance_posterior(x):
            pred=1
        else:
            pred=0
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for sample in test_set:
        if map_classifier.predict(sample)==sample[-1]:
            acc+=1
    acc=acc/len(test_set)   
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Add regularization to the covariance matrix

    x=x[:-1]
    
    dominator = 1 / np.sqrt(((2 * np.pi) ** (len(x))) * np.linalg.det(cov))
    
    top = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    
    pdf = dominator * np.exp(top)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        self.dataset=dataset
        self.class_value=class_value
        last_col = self.dataset[:, -1]  # Select the last column
        mask = last_col == self.class_value  # Create a boolean mask where the last column equals 1
        result = self.dataset[:,[0,1]][mask]  # Select the rows where the mask is True
        self.mean=np.mean(result,axis=0)
        self.cov=np.cov(result.T)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        last_col = self.dataset[:, -1]  # Select the last column
        mask = last_col == self.class_value  # Create a boolean mask where the last column equals 1
        result = self.dataset[mask]
        prior=len(result)/len(self.dataset)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihood = (multi_normal_pdf(x,self.mean,self.cov))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior=self.get_prior()*self.get_instance_likelihood(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0=ccd0
        self.ccd1=ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.ccd0.get_prior()>self.ccd1.get_prior():
            pred=0
        else:
            pred=1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0=ccd0
        self.ccd1=ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.ccd0.get_instance_likelihood(x)>self.ccd1.get_instance_likelihood(x):
            pred=0
        else:
            pred=1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.dataset=dataset
        self.class_value=class_value 
        self.NumberOfOptions =[]
        self.NumberOfSamplesOfClass=0
        self.NumberOfSamplesIntersection={}
        last_col = self.dataset[:, -1]  # Select the last column
        mask = last_col == self.class_value  # Create a boolean mask where the last column equals 1
        result = self.dataset[:,:-1][mask]  # Select the rows where the mask is True
        self.NumberOfSamplesOfClass=len(result)
        self.NumberOfOptions= [[(np.unique(result[:, i])),(np.unique(result[:, i])).size] for i in range(result.shape[1])]
        for i in range(len(result.T)):
            for x in (np.unique(result[:, i])):
                   self.NumberOfSamplesIntersection[(i,x)] = np.sum(result[:, i] == x)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        last_col = self.dataset[:, -1]  # Select the last column
        mask = last_col == self.class_value  # Create a boolean mask where the last column equals 1
        result = self.dataset[mask]
        prior=len(result)/len(self.dataset)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihoodlist = []

        for i in range(len(x)-1):
            if x[i] not in self.NumberOfOptions[i][0]:
                likelihoodlist.append(EPSILLON)
            else:
                like=(1+self.NumberOfSamplesIntersection[(i,x[i])])/(self.NumberOfOptions[i][1]+self.NumberOfSamplesOfClass)
                likelihoodlist.append(like)
        likelihood=np.prod(likelihoodlist)


        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior=self.get_prior()*self.get_instance_likelihood(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0=ccd0
        self.ccd1=ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.ccd0.get_instance_posterior(x)>self.ccd1.get_instance_posterior(x):
            pred=0
        else:
            pred=1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for sample in test_set:
            if sample[-1]==self.predict(sample):
                acc+=1
        acc=acc/len(test_set)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return acc


