import os
import tensorflow as tf
from load_data import load_and_preprocess_data
from model import create_model

def train_and_evaluate() -> None:
    """
    Load and preprocess the MNIST data, create and train a model, evaluate its performance, and save the trained model.
    
    DID THE TEST WORK
    """
    # Load and preprocess data
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

    # Create the model
    model = create_model()

    # Train the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')

    #check if the model directory exists
    if not os.path.exists('models/mnist_model'):
        os.makedirs('models/mnist_model')

    # Save the trained model
    model.save('models/mnist_model')

if __name__ == '__main__':
    train_and_evaluate()
