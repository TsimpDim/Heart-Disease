import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.python.data import Dataset
import os
from matplotlib import pyplot as plt
import math

def print_header(msg):
    print('\n' + '*' * 10 + f" {msg} " + '*' * 10 + '\n')

def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
            for my_feature in input_features])

def input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                            
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_linear_regressor_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets,
    l2_strength
    ):

    periods = 30
    steps_per_period = steps / periods

    # Create a linear regressor object
    optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l2_regularization_strength=l2_strength)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=optimizer
    )

    # Create input functions
    training_input_fn = lambda: input_fn(
        training_examples,
        training_targets["heart_disease"],
        batch_size=batch_size
    )

    predict_training_input_fn = lambda: input_fn(
        training_examples,
        training_targets["heart_disease"],
        num_epochs=1,
        shuffle=False
    )

    predict_validation_input_fn = lambda: input_fn(
        validation_examples,
        validation_targets["heart_disease"],
        num_epochs=1,
        shuffle=False
    )

    # Train the model
    print_header("Training Model")

    training_rmse = []
    validation_rmse = []

    for period in range(0, periods):
        
        # Train model for another iteration
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Compute predictions
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        # Validate
        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute and print loss
        training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_targets))
        
        print(f"Period {period}: {training_root_mean_squared_error} {validation_root_mean_squared_error}")

        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)

    print_header("Training Finished With")

    # Print hyper-parameters
    print(f"learning rate {learning_rate}")
    print(f"steps {steps}")
    print(f"batch_size {batch_size}")

    # Print weights
    print_header("Weights")
    print("".join(["\n" + f"{i} {linear_regressor.get_variable_value(i)}" for i in linear_regressor.get_variable_names()]))

    print(f"\n{optimizer.variables()}")

    # Print RMSE over time
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("RMSE over Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    return linear_regressor

def main():

    # Configure Tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR messages are not printed
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # Only show errors
    
    # Configure pandas
    pd.options.display.max_rows = 10
    pd.options.display.float_format = '{:.1f}'.format

    # Load dataset (DataFrame object)
    dataset = pd.read_csv("./heart.csv")

    # Re-index (randomize) dataset
    dataset = dataset.reindex(np.random.permutation(dataset.index))
    
    # Print sample data
    print_header("Sample Data")
    print(dataset)

    # Describe data
    print_header("Description")
    print(dataset.describe())

    # Define input features and training set
    training_examples = dataset.head(200)[
        [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]
    ]

    # Define output target and target set
    training_targets = pd.DataFrame()
    training_targets["heart_disease"] = (dataset.head(200)["target"])

    # Define validation set
    validation_targets = pd.DataFrame()
    validation_targets["heart_disease"] = (dataset.tail(100)["target"])

    validation_examples = dataset.tail(100)[
        [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]
    ]

    # Train model
    linear_regressor = train_linear_regressor_model(
        learning_rate=0.0000005,
        steps=500000,
        batch_size=10,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets,
        l2_strength=0.5
    )



if __name__ == "__main__":
    main()
