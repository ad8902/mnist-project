import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Check if model exists in the current directory first
if os.path.exists('mnist_model.h5'):
    model_path = 'mnist_model.h5'
elif os.path.exists('models/mnist_cnn.h5'):
    model_path = 'models/mnist_cnn.h5'
else:
    print("Model file not found! Please train the model first.")
    exit()

# Load the trained model
print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Function to preprocess your own images
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to MNIST dimensions
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255
    return img_array

# Test with a sample image
try:
    image_path = 'sample_digit.png'  # Create a 28x28 image of a digit
    processed_image = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    
    # Show results
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {prediction[0][predicted_digit]*100:.2f}%")
    
    # Display the image
    img = Image.open(image_path)
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted: {predicted_digit}')
    plt.show()
    
except FileNotFoundError:
    print("No sample image found. Testing with first test image...")
    
    # Load test data
    (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    test_image = test_images[0].reshape(1, 28, 28, 1).astype('float32') / 255
    test_label = test_labels[0]
    
    prediction = model.predict(test_image)
    predicted_digit = np.argmax(prediction)
    
    print(f"Actual digit: {test_label}")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {prediction[0][predicted_digit]*100:.2f}%")
    
    plt.imshow(test_images[0], cmap='gray')
    plt.title(f'Actual: {test_label}, Predicted: {predicted_digit}')
    plt.show()
    