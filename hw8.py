"""

Inputs are the selected options for different methods and questions.
Outputs are results for the selected questions.

"""

import numpy as np
import pandas as pd

from libsvm import svm
from libsvm import svmutil

from svmutil import *
from svm import *

from random import shuffle

def read_data():
    
    #   TRAINING DATASET
    
    #read training data from txt file put it into pandas dataframe
    training_dataset = pd.read_csv('features.train.txt', sep=" ", header=None)
    training_dataset.columns = ["NaN", "digit", "intensity", "symmetry"]
    
    
    #   TESTING DATASET
    
    #read testing data from txt file put it into pandas dataframe
    testing_dataset = pd.read_csv('features.test.txt', sep=" ", header=None)
    testing_dataset.columns = ["NaN", "digit", "intensity", "symmetry"]
    
    return training_dataset, testing_dataset
    

def sep_X_Y(training_dataset, testing_dataset):
    
    #   TRAINING DATASET
    
    #extracting training dataset features and labels
    X_train = training_dataset[["intensity", "symmetry"]] # features
    Y_train = training_dataset["digit"]   # labels           
    
    #convert pandas data frame to numpy array for features
    X_train_array = X_train.to_numpy()
    #convert string feature values to numbers
    X_train_array = X_train_array.astype(np.float)
    
    #convert pandas data frame to numpy array for class label
    Y_train_array = Y_train.to_numpy()
    #convert string label values to numbers
    Y_train_array = Y_train_array.astype(np.float)
    
    #   TESTING DATASET
    
    #extracting training dataset features and labels
    X_test = testing_dataset[["intensity", "symmetry"]] # features
    Y_test = testing_dataset["digit"]   # label
    
    #convert pandas data frame to numpy array for features
    X_test_array = X_test.to_numpy()
    #convert string feature values to numbers
    X_test_array = X_test_array.astype(np.float)
    
    #convert pandas data frame to numpy array for features
    Y_test_array = Y_test.to_numpy()
    #convert string feature values to numbers
    Y_test_array = Y_test_array.astype(np.float)
    
    return X_train_array, Y_train_array, X_test_array, Y_test_array
    
    
def one_vs_all(training_X, training_Y, Q, C):    
            
    #construct the parameters
    parameter = svm_parameter()
    #   kernel is polynomial
    parameter.kernel_type = POLY
    #take the penalty value
    parameter.C = C
    #take the degree value
    parameter.degree = Q
    # γ = 1
    parameter.gamma = 1 
    # r = 1
    parameter.coef0 = 1 
    
    #minimum Ein value
    min_Ein = 1
    
    #maximum Ein value
    max_Ein = 0  
    
    for i in range(10):
        
        new_training_Y = create_new_dataset(training_X, training_Y, i, 0, False)
        
        #construct the problem with label and feature vectors
        problem = svm_problem( new_training_Y,  training_X )
        
        #train the model
        model = svm_train(problem, parameter)
        
        #print the results
        print("Results of ", i, " versus all")
        
        #test the model
        #find accuracy and ( number of true predicted data points / number data points )
        plabel, paccuracy,pvals = svm_predict( new_training_Y, training_X, model )
        
        """
        Binary classification error is used to find Ein.
        Ein is calculated as (number of false predicted points / number data points )
        since accuracy is ( number of true predicted data points / number data points ) 
        Ein can be find as ( 100 - percentage of accuracy ) / 100
        """
        Ein = ( 100 - paccuracy[0] ) / 100
        
        #find min Ein and classifier for min Ein
        if(min_Ein > Ein):
            min_Ein = Ein
            classifier_min_Ein = i
            #calculate number of support vectors for the classifier which has min Ein
            classifier_min_Ein_sv = model.get_nr_sv()

        #find max Ein and classifier for max Ein
        if(max_Ein < Ein):
            max_Ein = Ein
            classifier_max_Ein = i 
            #calculate number of support vectors for the classifier which has max Ein
            classifier_max_Ein_sv = model.get_nr_sv()
        print("Ein =", Ein )
        
        print("")
      
    #print results for min and max Ein and their classifiers
    print(classifier_max_Ein, "versus all has the highest Ein value with", max_Ein, "  (2-a)")
    print(classifier_min_Ein, "versus all has the lowest Ein value with", min_Ein, "  (3-a)\n")
    
    #print number of support vectors and difference between them
    print("\nNumber of support vectors", classifier_max_Ein, "versus all classifier is", classifier_max_Ein_sv)
    print("Number of support vectors", classifier_min_Ein, "versus all classifier is", classifier_min_Ein_sv)
    print("\nDifference is", classifier_max_Ein_sv - classifier_min_Ein_sv, "  (4-c)")


#creating feature and target label vectors for "one vs one" and "one vs all" classification methods
#one vs one parameter is true if method is "one vs one" and vice versa 
def create_new_dataset(X, Y, label1, label2, is_one_vs_one):
    
    #new dataset which is obtained by updating feature vector (deleting some feature vectors which has ignored target labels)
    #this dataset is used only for one versus one classification
    new_X = []
    #new dataset which is obtained by updating target labels in the Y vector 
    new_Y = []
    
    #one vs one classification
    if(is_one_vs_one):
        
        for i in range(len(Y)):
            
            """
            take label1 as the one digit for classification and label2 is another digit for classification
            take one classification digit's target label as 1, take other classification digit's target labels as 0
            ignore other digits
            pass target labels and feature values to new dataset lists
            """
            
            if Y[i] == label1:
                new_Y.append(1)
                new_X.append(X[i])
            
            elif training_Y[i] == label2:
                new_Y.append(0) 
                new_X.append(X[i])
        
        return new_X, new_Y
    
    #one vs all classification
    else:
        
        for i in range(len(Y)):
            
            """
            take label1 as the digit for classification
            take it's target label as 1, take all other digit's target labels as 0
            pass target label values new_Y list which is target label vector 
            """
            
            if Y[i] == label1:
                new_Y.append(1)
            
            else:
                new_Y.append(0) 
                
        return new_Y
    
    
def one_vs_one(training_X, training_Y, testing_X, testing_Y, Q, C, label1, label2):
    
    """
    create dataset by updating target label values 
    if target label is 1 take it as 1, if it is 5 take it as 0, disgard remaining target labels
    """
    
    #training dataset
    new_training_X, new_training_Y = create_new_dataset(training_X, training_Y, label1, label2, True)  
    
    #testing dataset
    new_testing_X, new_testing_Y = create_new_dataset(testing_X, testing_Y, label1, label2, True)
            
    #construct the parameters
    parameter = svm_parameter()
    #   kernel is polynomial
    parameter.kernel_type = POLY
    # γ = 1
    parameter.gamma = 1 
    # r = 1
    parameter.coef0 = 1 
        
    #construct the problem with label and feature vectors
    problem = svm_problem(new_training_Y,  new_training_X )
    
    #set min_Ein to 1 to find minimum Ein value
    min_Ein = 1
    
    for c_value in C:   #calculate results for every C value in the list
        
        parameter.C = c_value   #set penalty value
        
        for q_value in Q:   #calculate results for every Q value in the list
                
            parameter.degree = q_value   #set degree value
                    
            print("\nC =", c_value, " ---  Q =", q_value)

            model = svm_train(problem, parameter )   #train the model
            
            print("\nNumber of support vectors: ", model.get_nr_sv() )
        
            print("\nTRAINING DATASET VALUES")   #predict label of training dataset
        
            plabel, paccuracy, pvals = svm_predict(new_training_Y, new_training_X, model)
        
            Ein = ( 100 - paccuracy[0] ) / 100   #calculate Ein by using binary classification error
            print("Ein =", Ein )
        
            #find min Ein and C value for min Ein
            if(min_Ein > Ein):
                min_Ein = Ein
                Cvalue_min_Ein = c_value
        
        
            #predict label of testing dataset
            print("\nTESTING DATASET VALUES")
                
            plabel, paccuracy, pvals = svm_predict(new_testing_Y, new_testing_X, model)
            
            Eout = ( 100 - paccuracy[0] ) / 100   #calculate Eout by using binary classification error
            print("Eout =", Eout )
        
        print("\n-----------")
        
    print("C value as ", Cvalue_min_Ein, " has the lowest Ein with ", min_Ein)
        

def cross_val(training_X, training_Y, Q, C):
    
    #create datasets with one vs one classification
    new_training_X, new_training_Y = create_new_dataset(training_X, training_Y, 1, 5, True)
    
    #construct the parameters
    parameter = svm_parameter()
    #   kernel is polynomial
    parameter.kernel_type = POLY
    # γ = 1
    parameter.gamma = 1 
    # r = 1
    parameter.coef0 = 1 
    # Q = 2
    parameter.degree = Q
    # set cross valiation parameter to True
    parameter.cross_validation = True
    # set number of k-fold to 10
    parameter.nr_fold = 10
    
    #construct the problem with label and feature vectors
    problem = svm_problem(new_training_Y, new_training_X )
        
    #keep number of selected times for every C value
    selected_Clist = [0, 0, 0, 0, 0]
    
    #keep total Ecv values for every C value to calcuate average Ecv values
    total_Ecv = [0, 0, 0, 0, 0]
    
    for i in range(100):        # 100 runs
        
        min_Ecv = 1     #find min Ecv
        
        #keep selected C value's index in c_index variable
        c_index = 0
        
        print(i+1)        #print results
        
        #calculate Ecv for every C value in the list
        for c_value in C:
            
            parameter.C = c_value       #set C value
            
            print("\nResults for C =", c_value)
            
            model = svm_train(problem, parameter )      #train the model
            
            Ecv = ( 100 - model ) / 100     #calculate Ecv
            
            # sum all Ecv values for every C value to calculate average Ecv
            total_Ecv[c_index] += Ecv
            
            #find minimum Ecv
            if(min_Ecv > Ecv):
                min_Ecv = Ecv
                #find C value which gives minimum Ecv in every run
                selected_C = c_value
                selected_C_index = c_index
               
            print("Ecv =", Ecv )
            print("")
            c_index += 1
            
        print("Min Ecv =", min_Ecv ) 
        
        print("Selected C value is", selected_C)
        
        #increase selected C value by one in each run (to find the most selected C value)
        selected_Clist[selected_C_index] += 1
          
        print("\n---------\n")
    
    #hold average Ecv value for every C in average_Ecv list
    average_Ecv = [0, 0, 0, 0, 0]
    
    #calculate average Ecv values
    for i in range(len(total_Ecv)):
        average_Ecv[i] = total_Ecv[i] / 100
        
    print("--------------------------------------------------")
    print("C", C)
    print("\nThe number of times penalty values were chosen:\n")
    
    most_often_selected = 0 
    
    #print nnumber of selection times for every C value
    for k in range(len(C)):
        
        #find most often selected C value
        if(most_often_selected < selected_Clist[k]):
            most_often_selected = selected_Clist[k]
            max_sel_C = C[k]
            average_Ecv_val = average_Ecv[k]
        print(C[k], "was chosen for", selected_Clist[k], "times\n")
        
    print("\nC = ", max_sel_C, "1 is selected most often.\n")
    
    print("Average Ecv value is", average_Ecv_val, "for C =", max_sel_C)
        
    
def rbf_kernel(training_X, training_Y, testing_X, testing_Y, C, label1, label2):

    """
    create dataset by updating target label values 
    if target label is 1 take it as 1, if it is 5 take it as 0, disgard remaining target labels
    """
    
    #training dataset
    new_training_X, new_training_Y = create_new_dataset(training_X, training_Y, label1, label2, True)  
    
    #testing dataset
    new_testing_X, new_testing_Y = create_new_dataset(testing_X, testing_Y, label1, label2, True)
    
    problem = svm_problem(new_training_Y, new_training_X )
    
    #construct the parameters
    parameter = svm_parameter()
    # γ = 1
    parameter.gamma = 1
    #since default value of kernel_type is RBF, there is no need to set it
    
    for c_value in C:
        
        parameter.C = c_value    
        print("C =", c_value, "--- Q =", Q)

        #train the model
        model = svm_train(problem, parameter )
                
        #predict label of training dataset
        print("\nTRAINING DATASET VALUES")
        
        #   get accuracy value
        plabel, paccuracy, pvals = svm_predict(new_training_Y, new_training_X, model)
        
        Ein = ( 100 - paccuracy[0] ) / 100      #calculate Ein
        print("Ein =", Ein )
        
        
        #predict label of testing dataset
        print("\nTESTING DATASET VALUES")
                
        #   get accuracy value
        plabel, paccuracy, pvals = svm_predict(new_testing_Y, new_testing_X, model)
        
        Eout = ( 100 - paccuracy[0] ) / 100         #calculate Eout
        print("Eout =", Eout )
        
        print("\n-----------\n")
    
    
        
if __name__ == '__main__':
    
    #reading datasets and passing them to pandas dataframe
    training_dataset, testing_dataset = read_data()
    
    #extracting target labels from dataset and passing features and labels to array lists    
    training_X, training_Y, testing_X, testing_Y = sep_X_Y(training_dataset, testing_dataset)
    
    
    method = ""
    while(method != "0"):
        
        print("==================================================================")
        print("\n1-Polynomial Kernels")
        print("2-Cross Validation")
        print("3-RBF Kernel")
        method = input("Choose an option for implemented methods (Print 0 for exit): ")
        
        if(method == "1"):
            print("\nA-Question 2-3-4")
            print("B-Question 5")
            print("C-Question 6")
            question = input("Choose an option for the questions: ")
            print("")
            
            if(question == "A"):
                #   QUESTION 2, 3, 4
                #defining parameters
                #Q is the degree of the polynomial 
                #C is the penalty to control overfitting
                Q = 2
                C = 0.01    
                one_vs_all(training_X, training_Y, Q, C)
                
            elif(question == "B"):
                #   QUESTION 5
                print("QUESTION 5\n")
                Q = [2]
                C = [0.001, 0.01, 0.1, 1]
                one_vs_one(training_X, training_Y, testing_X, testing_Y, Q, C, 1, 5)
            
            elif (question == "C"):
                #   QUESTION 6
                print("QUESTION 6\n")
                Q = [2, 5]
                C = [0.0001, 0.001, 0.01, 1]
                one_vs_one(training_X, training_Y, testing_X, testing_Y, Q, C, 1, 5)
                
            elif (question == "0"):
                break
                
            else:
                print("Invalid Option")
                
        elif (method == "2"):    
            #   QUESTON 7, 8
            print("QUESTION 7, 8\n")
            Q = 2
            C = [0.0001, 0.001, 0.01, 0.1, 1]
            cross_val(training_X, training_Y, Q, C)
   
        elif(method == "3"):
            #OUESTION 9, 10
            print("QUESTION 9, 10\n")
            Q = 2
            C = [0.01, 1, 100, 10000, 1000000]
            rbf_kernel(training_X, training_Y, testing_X, testing_Y, C, 1, 5)
            
        elif (method == "0"):
            break
        
        else:
            print("Invalid Option")
            
            
