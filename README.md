**Step 1: Process the Data**

Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.
  1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
    - What variable(s) are the target(s) for your model?
    - What variable(s) are the feature(s) for your model?

  2. Drop the EIN and NAME columns.
  3. Determine the number of unique values for each column.
  4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
  5. Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.
  6. Use pd.get_dummies() to encode categorical variables.
  7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
  8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

<br><br>
**Step 2: Compile, Train, and Evaluate the Model**

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
  1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
  2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
  3. Create the first hidden layer and choose an appropriate activation function.
  4. If necessary, add a second hidden layer with an appropriate activation function.
  5. Create an output layer with an appropriate activation function.
  6. Check the structure of the model.
  7. Compile and train the model.
  8. Create a callback that saves the model's weights every five epochs.
  9. Evaluate the model using the test data to determine the loss and accuracy.
  10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

<br><br>
**Step 3: Optimize the Model**

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:
  - Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
      - Dropping more or fewer columns.
      - Creating more bins for rare occurrences in columns.
      - Increasing or decreasing the number of values for each bin.
      - Add more neurons to a hidden layer.
      - Add more hidden layers.
      - Use different activation functions for the hidden layers.
      - Add or reduce the number of epochs to the training regimen.
   
Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

  1. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
  2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.
  3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
  4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
  5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

<br><br>
**Step 4: Write a Report on the Neural Network Model**

**Neural Network Model Analysis for Alphabet Soup Charity**

**Overview of the Analysis**

The purpose of this analysis is to develop a predictive model to assist Alphabet Soup in identifying applicants with the best chances of success if funded. By leveraging a deep learning model, specifically a neural network, we aim to classify whether an applicant is likely to succeed (IS_SUCCESSFUL) based on a variety of application features. The model is built using a dataset containing information about past applicants, their classifications, and success statuses.

The analysis involves data preprocessing, creating and optimizing a neural network model, and evaluating its performance to achieve the target predictive accuracy of over 75%.

**Results**

**Data Preprocessing**

1. What variable(s) are the target(s) for your model?

	•	The target variable is IS_SUCCESSFUL, a binary variable indicating whether an applicant was successful (1) or not (0).

2. What variable(s) are the features for your model?

    •	The features include all encoded columns derived from the dataset except the target column. These features include:
  
	  -	APPLICATION_TYPE_T3, CLASSIFICATION_C1200, ORGANIZATION_Co-operative, etc.
	  -	Numeric variables such as ASK_AMT.

3. What variable(s) should be removed from the input data because they are neither targets nor features?

	•	Non-beneficial columns such as EIN and NAME were removed as they do not contribute meaningfully to predicting success.

Compiling, Training, and Evaluating the Model

4. How many neurons, layers, and activation functions did you select for your neural network model, and why?

	•	The final model consists of:

    - Input Layer: 33 neurons (one for each input feature).
    - Four Hidden Layers:
    - Layer 1: 100 neurons, relu activation.
    - Layer 2: 50 neurons, LeakyReLU activation.
    - Layer 3: 20 neurons, LeakyReLU activation.
    - Layer 4: 10 neurons, LeakyReLU activation.
    - Output Layer: 1 neuron, sigmoid activation.
    - These parameters were selected to balance the model’s capacity to learn complex patterns and prevent overfitting.

6. Were you able to achieve the target model performance?

	•	No, the final model achieved an accuracy of ~72.1%, which is below the 75% target.

7. What steps did you take in your attempts to increase model performance?

	•	Several steps were taken to optimize the model:

   - Feature Engineering: Lowered the cutoff value for both application type and classification.
   - Adjusting Neurons and Layers: Experimented with additional hidden layers and varying numbers of neurons.
   - Activation Functions: Used LeakyReLU to address potential issues with dead neurons.
	 - Epochs and Batch Size: Adjusted epochs and batch sizes for better convergence during training.
  
**Summary**

**Overall Results**

The neural network model successfully identified patterns in the data but failed to reach the desired 75% accuracy. The most significant challenge was likely the imbalance in the target variable and the noise introduced by low-correlation features.

**Recommendation for Improvement**

A different model, such as Random Forest, could provide better results due to its ability to handle complex relationships and feature importance. Unlike neural networks, Random Forests are less prone to overfitting when tuned correctly and can handle categorical and numeric data effectively. By applying Random Forest, we can gain insights into the relative importance of features, which might inform better feature engineering for future neural network attempts.

In conclusion, while the deep learning model provided reasonable results, a tree-based algorithm or ensemble learning approach might be better suited for this classification problem.

Utilized Xpert Learning Assistant and Chat GPT for assistance in coding and resolving errors.
