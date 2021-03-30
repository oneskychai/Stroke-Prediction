This project is a stroke prediction model built from a data set with 5110 entries and 10 predictors
The data was downloaded from https://www.kaggle.com/fedesoriano/stroke-prediction-dataset/download
The data is first explored and analyzed to reveal trends
The data is then divided into training and test sets for model building
Several models are built and tested using the ROC curve as a metric
The best model is tested against a simulated validation set comprised of random samples from the data
This is done because the dataset is too small to further partition for training and testing

stroke_analysis.R contains all code for this project