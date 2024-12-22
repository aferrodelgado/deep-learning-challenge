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
