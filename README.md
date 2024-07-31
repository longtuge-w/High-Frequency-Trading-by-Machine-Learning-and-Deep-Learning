# High-Frequency-Trading-by-Machine-Learning-and-Deep-Learning

This project aims to predict future strategies by applying both machine learning and deep learning techniques. It leverages a variety of models including LSTM (Long Short-Term Memory), ALSTM (Attention-based LSTM), TCN (Temporal Convolutional Networks), GATS (Graph Attention Networks), and SFM (State Frequency Memory) to analyze and predict based on time-series data.

## Structure

The project is organized into several Python scripts, each serving a specific purpose:

- `utils.py`: Contains utility functions for preprocessing and manipulating data.
- `LSTM.py`: Defines the LSTM model for time-series prediction.
- `ALSTM.py`: Implements the ALSTM model, which enhances LSTM with an attention mechanism.
- `TCN.py`: Contains the Temporal Convolutional Network model, suitable for sequence modeling tasks.
- `GATS.py`: Defines the Graph Attention Network model, leveraging attention mechanisms for graph-structured data.
- `SFM.py`: Implements the State Frequency Memory model for capturing multi-frequency patterns in time-series.
- `ML.py`: Provides implementations of various machine learning models for comparison and ensemble learning.
- `DL.py`: Central script that orchestrates the data loading, model training, evaluation, and prediction tasks using deep learning models.

## Features

- **Preprocessing**: Standardizes and transforms the input data to be model-ready.
- **Model Implementation**: Offers a diverse set of models to tackle time-series prediction from different angles.
- **Evaluation and Prediction**: Supports model training, performance evaluation, and future strategy predictions.
- **Extensibility**: Easily extendable framework to include more models or adapt existing ones to new datasets.

## Usage

To use this framework for your prediction tasks, follow these steps:

1. Prepare your dataset according to the format expected by the `utils.py` preprocessing functions.
2. Choose the model(s) you wish to train from the list of available Python scripts (e.g., `LSTM.py`, `ALSTM.py`).
3. Use the `DL.py` script to train and evaluate the model on your dataset. This script will automatically handle data loading, model initialization, training, and evaluation.
4. For machine learning models, use the `ML.py` script to perform similar tasks with classical machine learning approaches.

## Dependencies

This project requires the following Python libraries:
- NumPy
- Pandas
- PyTorch
- scikit-learn
- Matplotlib (for data visualization)
- tqdm (for progress bars)

Ensure these are installed in your Python environment before running the scripts.

## Contribution

Feel free to contribute to this project by suggesting improvements, adding new models, or optimizing existing code.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
