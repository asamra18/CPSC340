import os
import pickle
import gzip
import argparse
import numpy as np
import platform
from sklearn.preprocessing import LabelBinarizer
from KNN import KNN
from sklearn.model_selection import cross_val_score
import Regression
from Regression import logRegL2, kernelLogRegL2


def load_dataset(filename):
    with open(os.path.join('..', 'data', filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
   # parser.add_argument('-q', '--question', required=True)

   # io_args = parser.parse_args()
    question = '1.2'# io_args.question

    if question == "1.1":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)
        # By inspection I know that the number of examples will be 50,000
        # This allows me to set an upper limit on the number of neighbours I want to
        # Test For. I inspected the 'X' matrix using the debugger tool in PYCHARM

        k_values = np.array(range(1,20)) #np.array(range(1,10)) # range of K values I want to test. This is okay for my computers memory capabilities


        y_pred = np.zeros(k_values.size)
        best_k = 0 #This is a dummy variable I will use to keep track of the k value which returns the best test error
        best_error = 1000000000000000000 #Initalizing a random value to store the best test error and track the value of it

        for k in k_values:
            model = KNN(k)
            model.fit(X,y)

            #Computing the validation Error with Xtest and yest
            y_pred = model.predict(Xtest)
            test_error = np.mean(y_pred != ytest)
            if test_error < best_error:
                best_k = k
                best_error = test_error

        print(best_k)
        print(best_error)

    elif question == "1.2":

        with gzip.open(os.path.join('mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")


        X, y = train_set
        X = X.astype(np.float32)
        y = y.astype(np.float32)


        Xtest, ytest = test_set
        Xtest = Xtest.astype(np.float32)
        ytest = ytest.astype(np.float32)




        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        p_vals = np.array([1,2]) #This is teh array of P values I wish to compute a poly regresion with

        #These are the hyperparameters I will finetune the RBF regression with
        sigArray = [10 ** -2, 10 ** 1, 1, 10 ** 1, 10 ** 2]
        lamArray = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1]
        trainError = 1000000000000000
        BestError = 10000000000000000
        bestP = 0

        #The loop below is finding the optimal p for the logistic regression
        #The code used is similar to assignment 5

        for p in p_vals:
            poly_kernel = kernelLogRegL2(kernel_fun=Regression.kernel_poly, p = p)
            poly_kernel.fit(X, y)
            test_Error = np.mean(poly_kernel.predict(Xtest) != ytest)
            if test_Error < BestError:
                BestError = test_Error
                bestP = p
        print(bestP)
        print(BestError)














    else:
        print("Unknown question: %s" % question)