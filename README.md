# Random Forest Classifier on MNIST Data

This project demonstrates the application of a Random Forest Classifier on the MNIST dataset to classify handwritten digits (0-9). The MNIST dataset is a well-known dataset in the machine learning community, consisting of 28x28 pixel grayscale images of handwritten digits.

## Project Overview
___Data Loading and Preprocessing___
<br>
___Model Training___
<br>
___Model Evaluation___
<br>
___Hyperparameter Tuning___
<br>
___Model Persistence___
<br>
### 1. Data Loading and Preprocessing
#### Importing Libraries
We start by importing the necessary libraries for our project. This includes libraries for data handling, machine learning, and visualization.
#### Loading the MNIST Dataset
The MNIST dataset is loaded using TensorFlow's 'keras' module.
#### Normalizing the Pixel Values
The pixel values are normalized to a range of 0 to 1 by dividing by 255.0.
#### Reshaping and Combining Data
The data is reshaped and combined to prepare for the training-test split.

### 2. Model Training
#### Train-Test Split
The data is split into training and testing sets.
#### Defining the Random Forest Model
A Random Forest model is defined with specific parameters.
#### Fitting the Model
The model is trained on the training data.

### 3. Model Evaluation
#### Predictions and Accuracy
The model's predictions and accuracy on the test dataset are evaluated.
#### Confusion Matrix
A confusion matrix is generated to visualize the model's performance.

### 4. Hyperparameter Tuning
#### Grid Search
Grid Search is performed to find the best hyperparameters for the Random Forest model.
#### Randomized Search
Randomized Search is used as a computationally efficient alternative to Grid Search.

### 5. Model Persistence
#### Saving the Best Model
The best model obtained from Randomized Search is saved for later use.


## Conclusion
This project demonstrates a comprehensive approach to training, evaluating, and optimizing a Random Forest classifier on the MNIST dataset. The model achieves high accuracy and the process includes both Grid Search and Randomized Search for hyperparameter tuning.

## Repository Structure
___'notebook.ipynb'___: The main Jupyter notebook with all the code and outputs.
<br>
___'best_random_forest_model_for_mnist_data.pkl'___: The saved Random Forest model.
<br>
## Requirements
___Python 3.x___
<br>
___TensorFlow___
<br>
___scikit-learn___
<br>
___numpy___
<br>
___matplotlib___
<br>
___seaborn___
<br>
___joblib___
<br>
## How to Run
___1. Clone the repository.___
<br>
___2. Install the required libraries.___
<br>
___3. Run the Jupyter notebook.___
<br>
## Author
This project was created by Nihad Aslanzade.

## License
This project is licensed under the MIT License.
