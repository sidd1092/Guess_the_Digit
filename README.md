# Guess the DIGIT

## Overview

"Guess the DIGIT" is a Python-based project that uses a Convolutional Neural Network (CNN) to recognize handwritten digits. The project includes a simple graphical user interface (GUI) built with Pygame, where users can draw digits and get real-time predictions from the trained model.

## Features

- **Handwritten Digit Recognition**: Draw digits using the mouse, and the model predicts the digit in real-time.
- **User Interface**: Simple and intuitive UI using Pygame for drawing and displaying predictions.
- **CNN Model**: A Convolutional Neural Network (CNN) trained on the MNIST dataset to achieve high accuracy in digit recognition.

## Requirements

- Python 3.x
- Pygame
- TensorFlow
- Keras
- OpenCV
- NumPy

## Installation


## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/sidd1092/Guess_the_Digit

2. **Install the required packages**:
   ```
   pip install pygame tensorflow keras opencv-python numpy
   ```

3. **Download the MNIST dataset** (handled automatically in the code).

4. **Train the model** (if not using the pre-trained model):
   - Run the model training script to train the CNN on the MNIST dataset and save the model:
     ```python
     model_training_script.py
     ```

5. **Run the application**:
   - Execute the UI script to start the handwritten digit recognition interface:
     ```python
     digit_recognition_ui.py
     ```

## File Structure

- `digit_recognition_ui.py`: Contains the Pygame-based GUI for drawing digits and predicting them.
- `model_training_script.py`: Contains the code for training the CNN model on the MNIST dataset.
- `model.h5`: The pre-trained CNN model file (if included).

## Usage

1. **Run the Application**:
   ```
   python digit_recognition_ui.py
   ```

2. **Draw Digits**:
   - Use the mouse to draw digits on the window.
   - The model will predict the digit in real-time and display the result.

3. **Clear the Screen**:
   - Press the 'n' key to clear the drawing area and start fresh.

## Model Training

The model training script (`model_training_script.py`) uses the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits.

### Steps for Training:

1. **Load the Data**:
   - The script loads the MNIST dataset and splits it into training and testing sets.

2. **Preprocess the Data**:
   - The images are reshaped to fit the model input requirements and one-hot encoded for the labels.

3. **Build the Model**:
   - A sequential CNN model is built using Keras with two convolutional layers, a max-pooling layer, a flattening layer, and a dense output layer with softmax activation.

4. **Train the Model**:
   - The model is trained using the Adam optimizer and categorical cross-entropy loss function for 10 epochs.

5. **Save the Model**:
   - The trained model is saved to a file (`model.h5`) for later use in the UI application.

## Conclusion

"Guess the DIGIT" is a fun and interactive project that showcases the power of CNNs in recognizing handwritten digits. The project combines deep learning with a simple Pygame interface to provide an engaging user experience.
