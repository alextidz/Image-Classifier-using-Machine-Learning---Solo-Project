# Image Classifier using Machine Learning - Solo Project

## Overview 

### Main Code
This code was developed for a university project in order to build a MLP (Multi-Layer Perceptron) classifier using a large dataset of geographical/nature-related images. These images were originally categorised into classes such as "forest", "river", "sea lake", etc., and the purpose of the MLP classifier would be to correctly assign a given image into one of these catgories (i.e, when given an image of a river, it correctly classifies it as "river"). The original large dataset of these images were processed, standardised, and split into training and testing subsets using the holdout method (a 70/30 split). This meant that 70% of the data would be used during the training phase of the MLP classifier, and 30% of the data would be used to test this model's accuracy.

Whilst trying to develop the most accurate model possible, a variety of different values for different hyperparamaters such as hidden layer size, alpha, and max iterations were tested, allowing us to explore how each of these affected the overall perfomance of the MLP classifier. The values of these hyperparameters for the combination that resulted in the highest accuracy score after testing were returned and used in the final model.

A hypothesis test was then developed to determine if stratified cross-validation has an impact on the performance of an MLP classifier when compared with non-stratified cross-validation. A model was trained and eveluated using both 5-fold StratifiedKFold and 5-fold KFold, comparing the overall accuracy for both using a paired t-test. The result of this test then determined if the performance was affected by the use of stratified cross-validation. 

Locally Linear Embedding was also brifely used to obtain a 2-dimensional representation of all samples in the dataset.

### Testing Code
In order to run and test each of the functions in the main code, a test code file was developed in Jupyter Notebook. This code ran the functions in the main code as appropriate, outputted relevant return data, and provided appropriate visual representations of data where required or necessary. These visual representations include:
- The affect of standardisation on the image pixel data
- The loss, training accuracy and testing accuracy accross iterations when developing the MLP classifier
- The impact of different alpha values on the accuracy, weights and biases of a model
- A comparison of the accuracy of a model when using stratified and non-stratfied cross-validation
- The seperability of classes in a 2-dimensional space obtained using locally linear embedding 

## Key Features
This project contains the following functionality:

- Pre-processing of image data into lower-resolution, flattened, 1-dimensional arrays
- Standardisation of this data
- Development of a MLP classifier, training and testing using a variety of combinations of values for different hyperparamaters to obtain the most accurate model
- Experemintation of the affect of hyperparameters on an MLP classifier's accuracy
- Hypothesis test to determine if stratified cross-validation impacts MLP classifier performance compared to non-stratfied cross-validation
- Locally Linear Embedding to obtain a 2-dimensional representation of all samples in the dataset
- Relevant visual representations of data throughout
  
## Technologies Used
This project uses the following technologies:

- Python - For the development of the functions written in the main code
- Jupyter Notebook - For running and testing the functions written in the main code, and providing corresponding data visualisations 

## Project Structure 
This broken down into the following:

- Main Python code - "coc131_cw.py"
- Jupyter Notebook test code - "test1.ipynb"
- Visualisations - list of png files with names corresponding to appropriate function

## Note
Due to the intensity of running this code, which required large numbers of iterations on an extremely large data set, it was not possible or realistic for me to test as many combinations of hyperparameter values as I would've liked in order to obtain a more accurate model. This was due to a combination of slow run time on this device, paired with a limited prioject time frame. Additionally, the original image dataset has not been included in this repository as it is far too large and contains tens of thousands of images.
