# Agriculture-Chatbot-Using-NLP
Agriculture chatbot using natural language processing(NLP) and artificial neural network(ANN).
Certainly! Here’s a detailed explanation of the provided code, which is a script for a chatbot using natural language processing (NLP) and machine learning. The script includes training models for intent classification and entity extraction, as well as running a loop to interact with users.

### Overview

1. **Imports and Dependencies**:
   - The script imports libraries for data manipulation (`pandas`, `numpy`), machine learning (`sklearn`, `keras`, `tensorflow`), and text processing (`re`, `nltk`).
   - The `keras` and `tensorflow` libraries are used for building and managing neural network models.

### Function Definitions

#### 1. `trainIntentModel()`
   - **Purpose**: To train a neural network model for classifying user intents based on their queries.
   - **Steps**:
     1. **Data Loading**:
        - Loads a CSV file containing queries and their corresponding intents.
     2. **Text Preprocessing**:
        - Cleans and preprocesses queries by removing non-alphabetic characters, tokenizing, and applying stemming.
     3. **Feature Extraction**:
        - Converts the text data into a bag-of-words representation using `CountVectorizer`.
        - Saves the vectorizer state for later use.
     4. **Label Encoding**:
        - Encodes intents into numerical categories and applies one-hot encoding.
        - Saves a mapping of intents to numeric labels.
     5. **Model Training**:
        - Defines and trains a Sequential neural network with dense layers for classification.
        - Saves the trained model.

#### 2. `trainEntityModel()`
   - **Purpose**: To train a model for recognizing entities in text.
   - **Steps**:
     1. **Data Loading**:
        - Loads a CSV file containing words and their associated labels.
     2. **Text Preprocessing**:
        - Applies stemming to words.
     3. **Feature Extraction**:
        - Converts words into a bag-of-words representation using `CountVectorizer`.
        - Saves the vectorizer state for later use.
     4. **Label Encoding**:
        - Encodes entity labels into numerical categories.
        - Saves a mapping of entity labels to numeric values.
     5. **Model Training**:
        - Trains a Gaussian Naive Bayes model for entity classification.
        - Saves the trained model.

#### 3. `getEntities(query)`
   - **Purpose**: To extract entities from a given query using the trained entity model.
   - **Steps**:
     1. **Feature Extraction**:
        - Transforms the input query into the feature space used during training.
     2. **Prediction**:
        - Uses the trained entity model to predict entity labels.
     3. **Entity Mapping**:
        - Maps predicted labels back to their corresponding entity names using the saved label mapping.

### Main Execution Loop

1. **Loading Models and Vectorizers**:
   - Loads the saved intent classification model and entity model, along with their vectorizers.

2. **User Interaction Loop**:
   - Continuously takes user input and processes it:
     1. **Text Preprocessing**:
        - Cleans and preprocesses user input (e.g., removing non-alphabetic characters, stemming).
     2. **Intent Prediction**:
        - Transforms the preprocessed text and predicts the intent using the loaded model.
        - Maps the prediction back to the intent label.
        - Responds to the user based on the predicted intent.
     3. **Entity Extraction**:
        - Extracts entities from the user input using the trained entity model.
        - Creates a mapping between extracted entities and the original tokens.

### Summary

- **Training**: The script trains models to classify user intents and extract entities. It involves data loading, text preprocessing, feature extraction, and model training.
- **Interaction**: It then uses these models to interact with users, predict their intents, and extract entities from their queries.

### Issues Noticed

1. **Duplicate Code**: There are duplicated definitions of `trainIntentModel()` and `trainEntityModel()` functions. Only one definition of each should be kept.
2. **Unused Imports**: Some imports (e.g., `confusion_matrix` and `train_test_split`) are not used in the script.

Make sure you address these issues to clean up and optimize the code.