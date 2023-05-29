###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    X = (X-np.mean(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))
    y = (y-np.mean(y))/(np.max(y)-np.min(y))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    X = np.c_[np.ones(len(X)), X]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    m = X.shape[0]
    h = np.dot(X, theta)
    J = np.sum((h - y) ** 2) / (2 * m)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    m = X.shape[0]
    for i in range(num_iters):
        h = np.dot(X, theta)
        loss = h - y
        gradient = np.dot(X.T, loss) / m
        theta = theta - alpha * gradient
        J_history.append(compute_cost(X, y, theta))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    pinv = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    pinv_theta = np.dot(pinv, y)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    m = X.shape[0]
    for i in range(num_iters):
        h = np.dot(X, theta)
        loss = h - y
        gradient = np.dot(X.T, loss) / m
        theta = theta - alpha * gradient
        J_history.append(compute_cost(X, y, theta))
        if i > 0 and (J_history[i-1]-J_history[i]) < 1e-8:
            break


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    np.random.seed(42)
    theta = np.random.random(size=X_train.shape[1])
    for alpha in alphas:
        theta_new = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)[0]
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta_new)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    #####c######################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    nonSelected = [i+1 for i in range(X_train.shape[1])]
    X_train = apply_bias_trick(X_train)
    X_val = apply_bias_trick(X_val)
    selected_features.append(0)
    for j in range(5):
        np.random.seed(42)
        theta = np.random.random(size=len(selected_features)+1)
        features_dict = {}
        for i in nonSelected:
            selected_features.append(i)
            X_train_short = X_train[:, selected_features]
            X_val_short = X_val[:, selected_features]
            theta_new = efficient_gradient_descent(X_train_short, y_train, theta, best_alpha, iterations)[0]
            features_dict[i] = compute_cost(X_val_short, y_val, theta_new)
            selected_features.remove(i)
        best_feature = min(features_dict, key=features_dict.get)
        selected_features.append(best_feature)
        nonSelected.remove(best_feature)
    selected_features = [i-1 for i in selected_features]
    selected_features.remove(-1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    help_set = df.columns
    new_dic = {}
    for col1 in df.columns:
        for col2 in help_set:
            new_dic[col1+'*'+col2] = df[col1]*df[col2]
        help_set = help_set[1:]
    data = pd.DataFrame.from_dict(new_dic)
    df_poly = pd.concat([df_poly, data], axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly