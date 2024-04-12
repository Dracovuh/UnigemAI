1. Import Statements:
   - Imports necessary libraries and modules for the script.

2. Test Function:
   - `evaluation(model, x_test_reshaped, y_test)`: Function to evaluate the model's performance.
   - `prepare_training_data(training_datas_scaled, WS, max_length_token_data, feature_count)`: Prepares training data for the model.

3. Logger Setup:
   - Sets up logging configurations.

4. Constants:
   - `Config`: Class containing configurations for the script.
   - `INTERVALS`: List of intervals.
   - `Feature`: Class defining input and output features.

5. `training` Function:
   - Main function for model training.
   - Calls various sub-functions to process data, define the model, train the model, save the model, evaluate the model, and upload the model.

6. `define_model` Function:
   - Defines the LSTM model for training or loads a pre-trained model.

7. Model Training Functions:
   - `train_model(model, x_train, y_train, epoch)`: Trains the model.
   - `save_model(model, interval)`: Saves the trained model.
   - `upload_model(interval: str)`: Uploads the model to a server.

8. Data Processing Functions:
   - `data_process(data, feature_cols)`: Processes data for training with scaling.
   - `data_process_scaless(data, feature_cols)`: Processes data for training without scaling.

This TOC provides an overview of the structure and functionalities of the provided code.



1. **Importing Libraries**
    - Imports necessary libraries and modules required for data processing, model training, evaluation, and logging.

2. **Defining Evaluation Function**
    - Defines the `evaluation` function used to evaluate the trained model's performance.
    - The function plots predictions vs. actual values for each token and calculates evaluation metrics such as RMSE and R-square.

3. **Defining Data Preparation Functions**
    - Includes functions for preparing training data, such as `prepare_training_data`.
    - These functions handle data preprocessing, padding, and reshaping to prepare input data for model training.

4. **Setting up Logging**
    - Configures logging settings to facilitate debugging and tracking of program execution.
    - Uses a colored formatter for improved readability of log messages.

5. **Defining Constants**
    - Defines constants such as database credentials, API URLs, and training parameters.
    - Organizes these constants within the `Config` class for easy access and modification.

6. **Defining Classes**
    - Contains the `Config` class nested within, which encapsulates configuration parameters.
    - Defines the `Feature` class, which contains input and output features used in training.

7. **Main Training Function**
    - **training(interval)**
        - Entry point for training the model, accepts an interval parameter specifying the time interval.
        - Reads training data, preprocesses it, defines the model, evaluates it, and returns True if successful.
        - Logs errors and returns False if any exception occurs during the training process.
    
    - **define_model(interval:str, max_length_token_data, WS, feature_count, is_scale:bool=True)**
        - Defines the LSTM model architecture based on the specified interval and other parameters.
        - Loads a pre-trained model if available and modifies it, or creates a new model from scratch.
    
    - **train_model(model, x_train, y_train, epoch)**
        - Trains the specified model using the provided training data for the specified number of epochs.
        - Returns training history containing loss and accuracy metrics.

    - **save_model(model, interval)**
        - Saves the trained model to a file in the specified interval directory.

    - **upload_model(interval: str)**
        - Uploads the saved model file to an API endpoint, specified by the interval.
        - Requires authentication token for authorization.

8. **Utility Functions**
    - Contains additional utility functions used in data processing, such as `data_process` and `data_process_scaless`.
    - These functions handle data loading, preprocessing, and scaling, preparing them for training.