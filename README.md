# VIT-Placement-Predictor

# 🎓 Student Placement Predictor Model

This project implements a **Neural Network** built with **TensorFlow Keras** to predict the placement status of a student (`Placement got`). The model is designed to handle a mix of numerical and categorical features efficiently using Keras Preprocessing Layers.

## 🚀 Key Features

  * **Multi-Input Handling:** Utilizes the Keras Functional API to accept separate inputs for different data types (CGPA, VIT Branch, Subject Branch).
  * **Built-in Preprocessing:** Integrates `Normalization` for numerical data and `StringLookup` and `CategoryEncoding` for categorical data directly into the model architecture.
  * **Binary Classification:** Uses a Sigmoid output layer and `binary_crossentropy` loss to predict a probability (0 to 1) of the student receiving a placement.

## 💻 Setup and Requirements

The code assumes you have a pandas DataFrame named `df` loaded into your environment, containing the columns specified below.

### Prerequisites

You need the following libraries installed:
  pip install numpy tensorflow scikit-learn pandas

### Data Requirements

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **Placement got** | Integer/Boolean | The target variable (e.g., 1 for placed, 0 for not placed). |
| **CGPA** | Float/Numeric | The student's Cumulative Grade Point Average. |
| **VIT Branch** | String/Object | The branch/department associated with the student's VIT degree. |
| **Subject Branch** | String/Object | The specific subject/specialization branch. |


## 🧠 Model Architecture & Processing

The model is structured using the Keras Functional API, allowing for distinct processing paths for each input before merging them.

### Input Streams

1.  **CGPA Input (Numerical)**
      * **Layer:** `Normalization`
      * **Function:** Standardises the CGPA values.
2.  **VIT Branch Input (Categorical)**
      * **Layer:** `StringLookup` to `CategoryEncoding`
      * **Function:** Converts branch names into unique integer IDs, and then into a **one-hot encoded** vector suitable for the neural network.
3.  **Subject Branch Input (Categorical)**
      * **Layer:** `StringLookup` to `CategoryEncoding`
      * **Function:** Performs the same one-hot encoding process as the VIT Branch input.

### Dense Layers (Deep Learning Core)

The processed features are concatenated and fed into a series of fully connected layers:

  * **Hidden Layers:** 128 to 64 to 32 neurons, all using **ReLU** activation.
  * **Regularization:** `Dropout(0.2)` is applied after the first two hidden layers to prevent overfitting.
  * **Output Layer:** `Dense(1)` with a **Sigmoid** activation, outputting the final placement probability.

## ⚙️ How to Run the Code

1.  **Load Data:** Ensure your `df` DataFrame is loaded and correctly structured.
2.  **Run Script:** Execute the Python script.

### Training Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Optimizer** | `tf.keras.optimizers.Adam` | Efficient gradient descent optimization. |
| **Learning Rate** | `0.0005` | A small rate for stable training. |
| **Loss Function** | `binary_crossentropy` | Standard for binary classification. |
| **Metrics** | `accuracy` | Tracks the correct prediction rate. |
| **Epochs** | `10` | The number of times the model sees the entire training set. |
| **Batch Size** | `256` | Number of samples processed before updating weights. |
| **Validation Split**| `0.1` | 10% of the training data used for internal validation during training. |

### Evaluation Output

The script will print the final performance metrics on the held-out test set.

## 💾 Saving the Model

After successful training, the final lines of the code show how to save the entire model (including the preprocessing layers) for later use without needing to retrain.
