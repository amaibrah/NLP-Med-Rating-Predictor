# Using NLP to Analyze Drug Reviews

The Drug Review Dataset was obtained from the UCI Machine Learning Repository. It contains patient reviews on specific drugs used to treat various conditions accompanied with a 10 star rating scale reflecting patient satisfaction. Each review also has a variable that measures the number of people who found the review useful. The data was already split 75%/25% into a training set and a testing set respectively with 6 features.

*The ultimate goal of this project is to build a model that is able to predict patient satisfaction ratings based on patient reviews using natural language processing, NLP.*

## Data cleaning and preparation
#### Features and datatypes:  
* drugName, string  
* condition, string
* review, string
* rating, float
* date, string
* usefulCount, integer

#### Null values:
* All 866 null values were within Condition column which will not impact predictive modeling
* Turned the null values to a new category within Condition called Unknown. Some medications are used for multiple conditions or used to treat only symptoms with no particular diagnosis. Some people who left a review might have also not wanted to put down a condition.

#### Additional cleaning:
* replace any strings containing the word span with Unknown

#### Additional data prep:
* Word cloud was created to see the most used words among all the reviews.
* Split rating column into 2 groups:
    * Ratings 5 and up are "good"
    * Ratings below 5 are "bad"
* Define predictor and target variable:
    * Predictor = reviews
    * Target variable = rating

## Text preprocessing
#### Text preprocessing using Spacy:
* Use en_core_web_lg for Spacy's largest English model
* Add commonly used words from word cloud that do not carry sentiment to stop words list
* Tokenize, remove stop words, remove punctuation, and lemmatize with spacy.token attributes

#### Vectorize cleaned test data
* Use TF-IDF Vectorizer:
* fit to only training data


## Build Model
#### Use RandomForest to build a model:
* GridSearchCV was used to look for best hyperparameters with 5-fold corss validation
* Apply best hyperparameters to RandomForest and fit to training data
* Predict using test set

#### Check performance of model on test data:
* Check confusion matrix
* Check model performance through precision and recall since data was imbalanced



