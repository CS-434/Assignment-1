import numpy as np
import time

SIGMA = 3

def main():

    #############################################################
    # These first bits are just to help you develop your code
    # and have expected ouputs given. All asserts should pass.
    ############################################################

    
    # Load training and test data as numpy matrices 
    train_X, train_y, test_X = load_data()


    #######################################
    # Q9 Hyperparmeter Search
    #######################################

    # Search over possible settings of k
    print("Performing 4-fold cross validation")
    for i in [0.001, 0.01]:
        t0 = time.time()
        SIGMA = i
        k = 99

        #######################################
        # TODO Compute train accuracy using whole set
        #######################################

        predictions = predict(train_X, train_y, train_X, k)

        train_acc = compute_accuracy(predictions, train_y)

        #######################################
        # TODO Compute 4-fold cross validation accuracy
        #######################################
        val_acc, val_acc_var = cross_validation(train_X, train_y, k=k)
        
        t1 = time.time()
        print("k = {:f} -- train acc = {:.2f}%  val acc = {:.2f}% ({:.4f})\t\t[exe_time = {:.2f}]".format(i, train_acc*100, val_acc*100, val_acc_var*100, t1-t0))
    
    #######################################


    #######################################
    # Q10 Kaggle Submission
    #######################################


    # TODO set your best k value and then run on the test set
    best_k = 99

    # Make predictions on test set
    pred_test_y = predict(train_X, train_y, test_X, best_k)    
    
    # add index and header then save to file
    test_out = np.concatenate((np.expand_dims(np.array(range(2000),dtype=int), axis=1), pred_test_y), axis=1)
    header = np.array([["id", "income"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')

######################################################################
# Q7 get_nearest_neighbors 
######################################################################
# Finds and returns the index of the k examples nearest to
# the query point. Here, nearest is defined as having the 
# lowest Euclidean distance. This function does the bulk of the
# computation in kNN. As described in the homework, you'll want
# to use efficient computation to get this done. Check out 
# the documentaiton for np.linalg.norm (with axis=1) and broadcasting
# in numpy. 
#
# Input: 
#   example_set --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   idx_of_nearest --   a k-by- list of indices for the nearest k
#                       neighbors of the query point
######################################################################

def get_nearest_neighbors(example_set, query, k):
    distances = np.linalg.norm(example_set - query, axis=1) # Computes row-wise norm
    k_vals = np.argpartition(distances, kth=k-1)[:k] # indices of k smallest -- unsorted

    if len(k_vals) > 1:
        idx_of_nearest = k_vals[np.argsort(k_vals)]
    else: idx_of_nearest = k_vals


    return idx_of_nearest, distances[idx_of_nearest]

######################################################################
# Q7 knn_classify_point 
######################################################################
# Runs a kNN classifier on the query point
#
# Input: 
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_label --   either 0 or 1 corresponding to the predicted
#                        class of the query based on the neighbors
######################################################################

def knn_classify_point(examples_X, examples_y, query, k):

    k_nearest, distances = get_nearest_neighbors(examples_X, query, k)

    sum = 0 
    weight_total = 0
    for i, item in enumerate(k_nearest):
        weight = np.exp(-(distances[i].item() ** 2)/SIGMA)
        weight_total += weight
        sum += examples_y[item].item() * weight

    predicted_label = round(sum / weight_total)

    return predicted_label

######################################################################
# Q8 cross_validation 
######################################################################
# Runs K-fold cross validation on our training data.
#
# Input: 
#   train_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   train_Y --  a n-by-1 vector of example class labels
#
# Output:
#   avg_val_acc --      the average validation accuracy across the folds
#   var_val_acc --      the variance of validation accuracy across the folds
######################################################################

def cross_validation(train_X, train_y, num_folds=4, k=1):

    subarrs_x = np.split(train_X, num_folds)
    subarrs_y = np.split(train_y, num_folds)
    predictions = []

    for i in range(num_folds):
        subtrain = []
        subquery = subarrs_x[i]
        labels = []

        for j in range(num_folds):
            subtrain = np.vstack([x for index, x in enumerate(subarrs_x) if index != i])
            labels = np.vstack([x for index, x in enumerate(subarrs_y) if index != i])

        predictions.extend(predict(subtrain, labels, subquery, k))

    
    avg_val_acc = compute_accuracy(np.vstack(predictions), train_y)
    var_val_acc = np.var(predictions == train_y)

    return avg_val_acc, var_val_acc



##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################


######################################################################
# compute_accuracy 
######################################################################
# Runs a kNN classifier on the query point
#
# Input: 
#   true_y --  a n-by-1 vector where each value corresponds to 
#              the true label of an example
#
#   predicted_y --  a n-by-1 vector where each value corresponds
#                to the predicted label of an example
#
# Output:
#   predicted_label --   the fraction of predicted labels that match 
#                        the true labels
######################################################################

def compute_accuracy(true_y, predicted_y):
    accuracy = np.mean(true_y == predicted_y)
    return accuracy

######################################################################
# Runs a kNN classifier on every query in a matrix of queries
#
# Input: 
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   queries_X --    a m-by-d matrix representing a set of queries 
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_y --   a m-by-1 vector of predicted class labels
######################################################################

def predict(examples_X, examples_y, queries_X, k): 
    # For each query, run a knn classifier
    predicted_y = [knn_classify_point(examples_X, examples_y, query, k) for query in queries_X]

    return np.array(predicted_y,dtype=int)[:,np.newaxis]

# Load data
def load_data():
    traindata = np.genfromtxt('train.csv', delimiter=',')[1:, 1:]
    train_X = traindata[:, :-1]
    train_y = traindata[:, -1]
    train_y = train_y[:,np.newaxis]
    
    test_X = np.genfromtxt('test_pub.csv', delimiter=',')[1:, 1:]

    return train_X, train_y, test_X


    
if __name__ == "__main__":
    main()
