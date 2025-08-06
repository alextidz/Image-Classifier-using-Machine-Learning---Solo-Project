import numpy as np
import os
from PIL import Image
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import ttest_rel

from sklearn.manifold import LocallyLinearEmbedding

# Please write the optimal hyperparameter values you obtain in the global variable 'optimal_hyperparm' below. This
# variable should contain the values when I look at your submission. I should not have to run your code to populate this
# variable.
optimal_hyperparam = {
    'hiddenLayerSize': (256, 128),
    'alpha': 1
}

class COC131:
    def q1(self, filename=None):
        """
        This function should be used to load the data. To speed-up processing in later steps, lower resolution of the
        image to 32*32. The folder names in the root directory of the dataset are the class names. After loading the
        dataset, you should save it into an instance variable self.x (for samples) and self.y (for labels). Both self.x
        and self.y should be numpy arrays of dtype float.

        :param filename: this is the name of an actual random image in the dataset. You don't need this to load the
        dataset. This is used by me for testing your implementation.
        :return res1: a one-dimensional numpy array containing the flattened low-resolution image in file 'filename'.
        Flatten the image in the row major order. The dtype for the array should be float.
        :return res2: a string containing the class name for the image in file 'filename'. This string should be same as
        one of the folder names in the originally shared dataset.
        """

        """
        AT: Function loads and processes the EuroSAT image dataset. Lowers resolution to 32 x 32 for all images, flattens
        all images into 1D arrays and stores them in self.x, and storels labels in self.y. If a filename is provided, also returns
        the original file paths and corresponding labels for visualisation during testing.
        """

        rootDirectory = 'EuroSAT_RGB'   

        data = []                       # Array for storing samples
        labels = []                     # Array for storing labels 

        classNames = sorted(os.listdir(rootDirectory))      # Sort class folders into alphabetical order

        for className in classNames:                        # Loop through each class folder                   

            classPath = os.path.join(rootDirectory, className)      # Create path from root directory to class folder
            if os.path.isdir(classPath) == False:                   # If path is not a directory, skip it                      
                continue

            images = os.listdir(classPath)      # List of all images in class folder

            for image in images:                # Loop through list of images

                imagePath = os.path.join(classPath, image)      # Create path from root directory to image file 
                imageFile = Path(imagePath)                     # Convert image path to Path object

                if imageFile.is_file() and imageFile.suffix.lower() in {'.png', '.jpg', '.jpeg'}:   # Check image is valid file and file type

                    resizedImage = Image.open(imagePath).resize((32, 32))   # Lower resolution of image to 32 x 32
                    imageArray = np.array(resizedImage).astype(float)       # Convert image to numpy array of type float   
                
                    data.append(imageArray)         # Add image array to data array
                    labels.append(className)        # Add image class label to labels array

        self.x = np.array(data, dtype=float)    # Store array of samples as numpy array of dtype float
        self.y = np.array(labels)               # Store array of labels as numpy array      


        if filename:                # If specific file name given

            for root, _, files in os.walk(rootDirectory):       # Loop through all subdirectories
                if filename in files:                           # If file is located
            
                    imagePath = os.path.join(root, filename)    # Create path from root directory to image file 
                    break
            else:                                                               # If file is not found      
                raise FileNotFoundError(str(filename) + ' not found in dataset.')    # Generate relevant error

            resizedImage = Image.open(imagePath).resize((32, 32))   # Lower resolution of image to 32 x 32
            imageArray = np.array(resizedImage).astype(float)       # Convert image to numpy array of type float   

            res1 = imageArray.flatten()                             # Flatten image
            res2 = os.path.basename(os.path.dirname(imagePath))     # Get class name from folder
        
        else:                       # If no file name given  

            res1 = np.zeros(1)      # Store empty placeholder array for flattened image
            res2 = ''               # Store empty class name
        
        return res1, res2



    def q2(self, inp):
        """
        This function should compute the standardized data from a given 'inp' data. The function should work for a
        dataset with any number of features.

        :param inp: an array from which the standardized data is to be computed.
        :return res2: a numpy array containing the standardized data with standard deviation of 2.5. The array should
        have the same dimensions as the original data
        :return res1: sklearn object used for standardization.
        """

        """
        AT: Function calculates mean and standard deviation of all input data, with values of 0 for standard deviation
        being replaced with a small number 1e-8 to avoid divide by 0 errors. It then standardises the data, resulting 
        in a mean of approximately 0 and a standard deviation of approximately 1. These are then scaled by a factor of 
        2.5, resulting in a standard deviation of approximately 2.5 for all features. Function returns an object of the
        original means and standard deviations used, and the final, standardised data.

        """

        means = np.mean(inp, axis = 0)                  # Calculate mean of each feature
        standardDeviations = np.std(inp, axis = 0)      # Calculate standard deviation of each feature

        for i in range(len(standardDeviations)):        # Iterate through standard deviations  

            if standardDeviations[i] == 0:              # If standard deviation is 0
                standardDeviations[i] = 1e-8            # Replace with small number to avoid divide by 0 errors

        standardisedData = (inp - means) / standardDeviations       # Standardise data using standardisation formula
    
        # All features will now have a mean of approximately 0 and a standard deviation of approximately 1

        res1 = {                        # Object used for standardisaton
            'means': means,
            'standardDeviations':  standardDeviations
        }

        res2 = standardisedData * 2.5   # Multiply standardised data by 2.5 

        # All features will now have a standard deviation of approximately 2.5

        return res1, res2



    def q3(self, test_size=None, pre_split_data=None, hyperparam=None):
        """
        This function should build a MLP Classifier using the dataset loaded in function 'q1' and evaluate model
        performance. You can assume that the function 'q1' has been called prior to calling this function. This function
        should support hyperparameter optimizations.

        :param test_size: the proportion of the dataset that should be reserved for testing. This should be a fraction
        between 0 and 1.
        :param pre_split_data: Can be used to provide data already split into training and testing.
        :param hyperparam: hyperparameter values to be tested during hyperparameter optimization.
        :return: The function should return 1 model object and 3 numpy arrays which contain the loss, training accuracy
        and testing accuracy after each training iteration for the best model you found.
        """

        """
        AT: Function splits data using a 70/30 train/test, standardises data and trains a MLPClassifier using different 
        combinations of alpha values and hidden layer sizes to find optimal hyperparameters, updating the global variable. 
        Function returns the best model, and numpy arrays for loss, training accuracy and testing accuracy.

        Due to issues with large runtimes on this device, two hyperparameters were tested, with three values selected for 
        each hyperparameter. These values were chosen based on experimentation and to give a balance between performance
        exploration and execution time.

        Other models used during testing resulted in higher values for training accuracy than the selected model, however
        they had a lower testing accuracy, indicating overfitting. The final model was therefore chosen due to its more accurate    
        classification of unseen data.
        """

        X = self.x.reshape(self.x.shape[0], -1)     # Flatten and store images
        y = self.y                                  # Store labels   

        # Split images and labels into training and testing groups (70/30)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0, stratify = y)

        standardisationObject, standardisedTrainingData = self.q2(X_train)  # Call q2 to standardise traning data  

        means = standardisationObject['means']                              # Original Means and Standard Deviations for training data                            
        standardDeviations = standardisationObject['standardDeviations']
        
        for i in range(len(standardDeviations)):        # Iterate through standard deviations  

            if standardDeviations[i] == 0:              # If standard deviation is 0
                standardDeviations[i] = 1e-8            # Replace with small number to avoid divide by 0 errors

        standardisedTestData = ((X_test - means) / standardDeviations) * 2.5    # Standardise test data using Means and Standard Deviations from training data

        hiddenLayerSizes = [(100,), (256, 128), (128, 64)]  # Hyperparameters
        alphas = [0.1, 1, 5]
        
        bestAccuracy = 0                # Variables to store best accuracy and optimal hyperparameters
        bestHiddenLayerSize = None
        bestAlpha = None

        for hiddenLayerSize in hiddenLayerSizes:        # Iterate through hidden layer sizes

            for alpha in alphas:                        # Iterate though alphas     

                # Create MLP model using hyperparameters
                model = MLPClassifier(hidden_layer_sizes = hiddenLayerSize, alpha = alpha, max_iter = 300, random_state =0)    
                model.fit(standardisedTrainingData, y_train)        # Train model using standardised training data set

                y_pred = model.predict(standardisedTestData)        # Predict class labels using standardised testing data set
                accuracy = accuracy_score(y_test, y_pred)           # Calculate accuracy of prediction

                if accuracy > bestAccuracy:         # Store the accuracy and hyperparameters of the models with the highest accruacy
                    bestAccuracy = accuracy
                    bestHiddenLayerSize = hiddenLayerSize
                    bestAlpha = alpha
        
        # Create new MLP model using these hyperparameters
        bestModel = MLPClassifier(hidden_layer_sizes = bestHiddenLayerSize, alpha = bestAlpha, max_iter = 1, warm_start = True, random_state = 0)    

        losses = []                 # Arrays to store losses, training accuracies and testing accuracies
        trainingAccuracies = []
        testingAccuracies = []

        for i in range(50):         # Train this model for 50 iterations   

            bestModel.fit(standardisedTrainingData, y_train)    # Train model using standardised training data set

            training_y_pred = bestModel.predict(standardisedTrainingData)   # Predict class labels using standardised training data set        
            trainingAccuracy = accuracy_score(y_train, training_y_pred)     # Calculate training accuracy

            testing_y_pred = bestModel.predict(standardisedTestData)        # Predict class labels using standardised testing data set 
            testingAccuracy = accuracy_score(y_test,testing_y_pred )        # Calculate testing accuracy

            losses.append(bestModel.loss_)                  # Store loss, training accuracy and test accuracy
            trainingAccuracies.append(trainingAccuracy)         
            testingAccuracies.append(testingAccuracy)

        
        global optimal_hyperparam       # Stores the optimal hyperparameters in global variable
        optimal_hyperparam = {
            'hiddenLayerSize': bestHiddenLayerSize,
            'alpha': bestAlpha
        }

        res1 = bestModel                        # Return best model and numpy arrays for loss, training accuracy and testing accuracy
        res2 = np.array(losses)                 
        res3 = np.array(trainingAccuracies)     
        res4 = np.array(testingAccuracies)      

        return res1, res2, res3, res4

        

    def q4(self):
        """
        This function should study the impact of alpha on the performance and parameters of the model. For each value of
        alpha in the list below, train a separate MLPClassifier from scratch. Other hyperparameters for the model can
        be set to the best values you found in 'q3'. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: res should be the data you visualized.
        """

        """
        AT: Function shows how different alpha values affect accuracy, weights and biases of an MLPClassifier. It trains    
        different models for each alpha value, using the optimal hyperparameters from q3. For each model, the weights, biases,
        and test accuracy are obtained. Function returns array of alpha values, array of weights, array of biases and array of
        accuracies to be used for visualisation.
        """

        X = self.x.reshape(self.x.shape[0], -1)     # Flatten and store images
        y = self.y                                  # Store labels

        # Split images and labels into training and testing groups (70/30)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0, stratify = y)

        standardisationObject, standardisedTrainingData = self.q2(X_train)  # Call q2 to standardise traning data  

        means = standardisationObject['means']                              # Original Means and Standard Deviations for training data                            
        standardDeviations = standardisationObject['standardDeviations']
        
        for i in range(len(standardDeviations)):        # Iterate through standard deviations  

            if standardDeviations[i] == 0:              # If standard deviation is 0
                standardDeviations[i] = 1e-8            # Replace with small number to avoid divide by 0 errors

        standardisedTestData = ((X_test - means) / standardDeviations) * 2.5    # Standardise test data using Means and Standard Deviations from training data

        alpha_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]    # Array of alpha values provided

        accuracies = []        # Arrays to store accuracy, weights and biases for each model
        weights = []           
        biases = []            

        for alpha in alpha_values:      # Iterate though alphas  

            # Create MLP model using optimal hidden layer size from q3
            model = MLPClassifier(hidden_layer_sizes = optimal_hyperparam['hiddenLayerSize'], alpha = alpha, max_iter = 300, random_state =0)    
            model.fit(standardisedTrainingData, y_train)        # Train model using standardised training data set

            y_pred = model.predict(standardisedTestData)        # Predict class labels using standardised testing data set
            accuracy = accuracy_score(y_test, y_pred)           # Calculate accuracy of prediction
            accuracies.append(accuracy)                         # Add accuracy to accuracies array

            weights.append(model.coefs_)            # Store weights (coefs_ is the list of arrays containng weights)  
            biases.append(model.intercepts_)        # Store biases (intercepts_ is the list of arrays containing the biases)

               
        res = {                             # Return all data to be visualised
            'alphas': alpha_values,         
            'accuracies': accuracies,       
            'weights': weights,             
            'biases': biases                
        }

        return res


    def q5(self):
        """
        This function should perform hypothesis testing to study the impact of using CV with and without Stratification
        on the performance of MLPClassifier. Set other model hyperparameters to the best values obtained in the previous
        questions. Use 5-fold cross validation for this question. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: The function should return 4 items - the final testing accuracy for both methods of CV, p-value of the
        test and a string representing the result of hypothesis testing. The string can have only two possible values -
        'Splitting method impacted performance' and 'Splitting method had no effect'.
        """

        """
        AT: Function performs hypothesis test to determine if stratified cross-validation has an impact on the performance 
        of an MLPClassifier when compared with non-stratified cross-validation. It trains and evaluates a model using both
        5-fold StratifiedKFold and 5-fold KFold, using the optimal hyperparameters from q3. It then compares the testing
        accuracies for both using a paired t-test. Function returns mean testing accuracies for both, the p-value from the test, 
        and the result of the test, indicating if performance was affected.
        """

        X = self.x.reshape(self.x.shape[0], -1)     # Flatten and store images
        y = self.y                                  # Store labels

        stratifiedKFold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)   # StratifiedKFold with 5 splits for stratified CV 
        kFold = KFold(n_splits =5, shuffle = True, random_state = 0)                        # KFold for non-stratified CV

        stratifiedAccuracies = []       # Array to store accuracies for stratified CV
        nonStratifiedAccuracies = []    # Array to store accuracies for non-stratified CV

        for train_index, test_index in stratifiedKFold.split(X, y):     # Iterate through stratified k fold
            
            X_train, X_test = X[train_index], X[test_index]     # Split data into training and testing sets
            y_train, y_test = y[train_index], y[test_index]

            standardisationObject, standardisedTrainingData = self.q2(X_train)  # Call q2 to standardise traning data  

            means = standardisationObject['means']                              # Original Means and Standard Deviations for training data                            
            standardDeviations = standardisationObject['standardDeviations']
            
            for i in range(len(standardDeviations)):        # Iterate through standard deviations  

                if standardDeviations[i] == 0:              # If standard deviation is 0
                    standardDeviations[i] = 1e-8            # Replace with small number to avoid divide by 0 errors

            standardisedTestData = ((X_test - means) / standardDeviations) * 2.5    # Standardise test data using Means and Standard Deviations from training data

            # Create MLP model using optimal hyperparameters from q3
            model = MLPClassifier(hidden_layer_sizes = optimal_hyperparam['hiddenLayerSize'], alpha = optimal_hyperparam['alpha'], max_iter = 300, random_state =0)
            model.fit(standardisedTrainingData, y_train)        # Train model using standardised training data set

            y_pred = model.predict(standardisedTestData)        # Predict class labels using standardised testing data set
            accuracy = accuracy_score(y_test, y_pred)           # Calculate accuracy of prediction
            stratifiedAccuracies.append(accuracy)               # Store accuracy for stratified fold in array for stratified accuracies


        for train_index, test_index in kFold.split(X, y):       # Iterate through non-stratified k fold

            X_train, X_test = X[train_index], X[test_index]     # Split data into training and testing sets
            y_train, y_test = y[train_index], y[test_index]

            standardisationObject, standardisedTrainingData = self.q2(X_train)  # Call q2 to standardise traning data  

            means = standardisationObject['means']                              # Original Means and Standard Deviations for training data                            
            standardDeviations = standardisationObject['standardDeviations']
            
            for i in range(len(standardDeviations)):        # Iterate through standard deviations  

                if standardDeviations[i] == 0:              # If standard deviation is 0
                    standardDeviations[i] = 1e-8            # Replace with small number to avoid divide by 0 errors

            standardisedTestData = ((X_test - means) / standardDeviations) * 2.5    # Standardise test data using Means and Standard Deviations from training data

            # Create MLP model using optimal hyperparameters from q3
            model = MLPClassifier(hidden_layer_sizes = optimal_hyperparam['hiddenLayerSize'], alpha = optimal_hyperparam['alpha'], max_iter = 300, random_state =0)
            model.fit(standardisedTrainingData, y_train)        # Train model using standardised training data set

            y_pred = model.predict(standardisedTestData)        # Predict class labels using standardised testing data set
            accuracy = accuracy_score(y_test, y_pred)           # Calculate accuracy of prediction
            nonStratifiedAccuracies.append(accuracy)            # Store accuracy for non-stratified fold in array for non-stratified accuracies


        _, p = ttest_rel(stratifiedAccuracies, nonStratifiedAccuracies)  # Run t-test to compare accuracies of CV methods

        if p < 0.05:      # If p < 0.05, splitting method impcated performance
            
            result = 'Splitting method impacted performance'

        else:           # If p < 0.05, splitting method had no effect

            result = 'Splitting method had no effect'


        res1 = np.mean(stratifiedAccuracies)        # Return final testing accuracies, p value and string result
        res2 = np.mean(nonStratifiedAccuracies)
        res3 = p
        res4 = result

        return res1, res2, res3, res4
    

    def q6(self):
        """
        This function should perform unsupervised learning using LocallyLinearEmbedding in Sklearn. You can assume that
        the function 'q1' has been called prior to calling this function.

        :return: The function should return the data you visualize.
        """

        """
        AT: Function applies Locally Linear Embedding to obtain a 2-dimensional representation of all samples in the dataset
        It was initially tested with n_neighbors = 10, but later increased to 25 after experimentation for a more local 
        structure and an improved class separability. Function returns the transformed 2D data along with the original class 
        labels, allowing for plotting of data points for visualisation.
        """

        X = self.x.reshape(self.x.shape[0], -1)     # Flatten and store images
        y = self.y                                  # Store labels

        # Create LLE model to transform data to 2D representation
        model = LocallyLinearEmbedding(n_components = 2, n_neighbors = 25, random_state =0)     # Set n_neighbors = 25, so each data point compared to 25 nearest neighbours
        transformedData = model.fit_transform(X)    # Transform data


        res = {                     # Return transformed data
            'data': transformedData,
            'labels': y
        }

        return res