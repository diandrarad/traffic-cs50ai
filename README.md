# Traffic Sign Recognition AI

### [Video Demo](https://youtu.be/EYIJz404cME)

## Description
This project showcases an artificial intelligence (AI) model that can accurately identify which traffic sign appears in a photograph. The model has been trained using the full GTSRB (German Traffic Sign Recognition Benchmark) dataset, which contains various traffic sign images. By leveraging deep learning techniques and computer vision, the AI model demonstrates remarkable accuracy in recognizing traffic signs, making it a valuable tool for applications such as autonomous vehicles and road safety systems.

## Technology
The Traffic Sign Recognition AI is built using the following technologies:

- Python: The programming language used for implementing the AI model and associated scripts.
- OpenCV-Python: A powerful computer vision library used for reading and preprocessing the traffic sign images.
- TensorFlow: An open-source machine learning framework used for building and training the deep learning model.
- Keras: A high-level neural networks API, running on top of TensorFlow, used for defining the model architecture and training the AI.
- GTSRB Dataset: The full GTSRB dataset, containing a wide range of traffic sign images, is utilized for training the AI model.
- scikit-learn: Simple and efficient tools for predictive data analysis.

## Functionality
The Traffic Sign Recognition AI performs the following tasks:

1. Data Loading: The provided load_data function reads the traffic sign images from the GTSRB dataset and prepares them for training. It utilizes OpenCV-Python to read each image, resizes them to a uniform size, and organizes them into image arrays and labels.
2. Model Architecture: The get_model function defines the architecture of the neural network used for traffic sign recognition. It employs convolutional and pooling layers to extract meaningful features from the images, followed by dense layers for classification. The model architecture can be customized and experimented with to optimize performance.
3. Training and Evaluation: The AI model is trained using the GTSRB dataset, where it learns to recognize different traffic signs. The model is then evaluated using test images to assess its accuracy and performance.
4. Traffic Sign Recognition: Once the AI model is trained, it can be used to identify which traffic sign appears in a photograph. By providing an input image, the model predicts the corresponding traffic sign label with high accuracy.

## What have been tried:

### 1. First Model:
333/333 - 2s - loss: 3.4948 - accuracy: 0.0553 - 2s/epoch - 7ms/step

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),  # Add dropout
            tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
        ]
    )

- Conv2D layer with 32 filters and (3, 3) kernel size, followed by ReLU activation
- MaxPooling2D layer with a pool size of (2, 2)
- Flatten layer to convert the output into a 1D tensor
- Dense layer with 64 units and ReLU activation
- Dropout layer with a rate of 0.2 to randomly drop 20% of the units
- Dense layer with NUM_CATEGORIES units (output layer) and softmax activation
- Adam optimizer with default learning rate (0.001)
- Categorical cross-entropy loss function
- Accuracy metric

### 2. Second Model:
333/333 - 3s - loss: 0.1420 - accuracy: 0.9689 - 3s/epoch - 9ms/step

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
        ]
    )

- Same as the first model, but with an additional Conv2D layer and MaxPooling2D layer before the Flatten layer
- Dropout layer with a rate of 0.5 to randomly drop 50% of the units
- Optimizer, loss function, and metrics are the same as the first model

### 3. Third Model:
333/333 - 4s - loss: 0.0448 - accuracy: 0.9885 - 4s/epoch - 11ms/step

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
        ]
    )

- Same as the last model, but with the addition of BatchNormalization layers after each Conv2D and Dense layer
- Optimizer, loss function, and metrics are the same as the third model

## What worked well in the second model:

- Increased model complexity: The second model has a more complex architecture with additional convolutional layers. This increased model capacity allows for better extraction of features from the input images, leading to improved accuracy.
- Increased depth: The deeper architecture of the second model allows for the extraction of more intricate patterns and higher-level representations from the input images. This depth helps capture more complex relationships in the data, which can lead to better performance.
- Dropout regularization: The addition of the dropout layer in the second model helps reduce overfitting by randomly deactivating a portion of the neurons during training. This regularization technique prevents the model from relying too heavily on specific features and encourages more robust learning.

## What didn't work well in the first model:

- Model capacity: The first model has a simpler architecture with fewer convolutional layers and a smaller dense layer. This limited model capacity may not be sufficient to capture and learn the intricate patterns present in the dataset, resulting in lower accuracy.

- Lack of regularization: The first model lacks explicit regularization techniques such as dropout. Without regularization, the model is more prone to overfitting, where it memorizes the training data instead of learning generalized patterns. This can lead to reduced performance on unseen data.

The third model with the additional BatchNormalization layers achieved better accuracy and lower loss compared to the second model. Here's an analysis of what worked well and what didn't:

## What worked well in the third model:

- Batch normalization: The addition of BatchNormalization layers in the second third helped in normalizing the activations between the layers. This normalization reduces the internal covariate shift and allows the model to learn more efficiently. It can lead to improved gradient flow, faster convergence, and better generalization, which likely contributed to the better accuracy and loss.
- Improved stability: The BatchNormalization layers help stabilize the learning process by reducing the impact of parameter initialization and the choice of learning rate. This stability can result in more consistent and reliable training, leading to better performance.
- Regularization: The second model also includes dropout and BatchNormalization, which work together to combat overfitting. Dropout randomly drops out units during training, reducing over-reliance on specific features. BatchNormalization provides regularization by reducing the sensitivity of the model to small changes in the input distribution.

## What didn't work well in the second model:

- Lack of normalization: The second model lacks BatchNormalization layers, which can result in internal covariate shift and slower convergence during training. Without normalization, the model may struggle to find an optimal set of weights and biases, leading to suboptimal accuracy and loss.
- Limited regularization: The second model only includes dropout as a regularization technique. While dropout is effective in preventing overfitting to some extent, the model may still benefit from additional regularization methods like BatchNormalization.

## What are been noticed:

- The second model with the additional convolutional layers and higher number of units in the dense layer achieved better accuracy and lower loss compared to the first model. The improvements in the second model, including increased model complexity, depth, and dropout regularization, likely contributed to its better performance in terms of accuracy and loss. The second model had a higher capacity to learn complex patterns and better control overfitting, resulting in improved performance compared to the simpler first model.
- The third model with the additional BatchNormalization layers achieved better accuracy and lower loss compared to the second model. The improvements in the third model, including the addition of BatchNormalization layers and the combination of dropout and BatchNormalization for regularization, likely contributed to its better performance in terms of accuracy and loss. The third model achieved better stability, normalization, and regularization, leading to improved training dynamics and enhanced generalization capabilities compared to the first model.

# Credits
This project was completed as part of the HarvardX CS50's Introduction to Artificial Intelligence with Python course. The import statements, global variable assignments, and main function were provided by the course, and the load_data and get_model functions were implemented by me.
