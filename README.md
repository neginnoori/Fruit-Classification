# FruitClassification
![img](https://github.com/neginnoori/fruitClassification/blob/main/fotor-ai-20240224142232.jpg)
The project aims to classify fruits (such as banana, apple, strawberry, and orange) into ripe and unripe categories using machine learning techniques. The code utilizes image processing and deep learning algorithms to accurately classify the fruit images. The repository includes the dataset used, pre-trained models, and Jupyter notebooks for training and evaluation. 
# Data 
This dataset was utilized from [`the FRINN GitHub`](https://github.com/ece324-2020/FRINN/tree/main) repository. The dataset comprises 15 subfolders containing training data in the form of PNG images. The images depict various types of fruits, such as apples, oranges, bananas, raspberries, and strawberries, each classified into three states: ripe, unripe, and rotten. The dataset serves as a valuable resource for training image recognition models, particularly in the realm of fruit classification algorithms.
# Methods
This repository showcases a method for fruit recognition using Convolutional Neural Networks (CNNs) and the VGG16 architecture. The primary objective is to develop a robust model capable of accurately classifying fruits into three categories: unripe, ripe, and rotten, based on their images.

Convolutional Neural Networks (CNNs) are powerful deep learning models specifically designed for image analysis tasks. They excel at automatically learning and extracting meaningful features from images, making them highly suitable for fruit recognition. CNNs consist of multiple layers, including convolutional layers that capture local patterns and features, pooling layers that downsample spatial dimensions, and fully connected layers that make predictions based on the extracted features.

In this project, we leverage the VGG16 architecture, a widely-used CNN architecture known for its simplicity and effectiveness in image classification tasks. VGG16 comprises 16 layers, including 13 convolutional layers and 3 fully connected layers. The convolutional layers employ small filters to capture intricate details, while pooling layers reduce spatial dimensions while retaining crucial features. The fully connected layers at the end of the network facilitate the final predictions based on the learned features.

The code provided in this repository implements the CNN model with the VGG16 architecture using the PyTorch deep learning framework. It encompasses data loaders to load and preprocess fruit images, a training loop that trains the model on the dataset, and an evaluation loop to assess the model's performance on a separate validation set. Additionally, the code incorporates early stopping to mitigate overfitting risks and includes functions to monitor and visualize the training progress.

By utilizing this repository, you can explore the potential of CNNs and the VGG16 architecture for fruit recognition tasks, fine-tune hyperparameters, and adapt the code to your own fruit image dataset. The trained model can find applications in various domains such as agriculture, food quality assessment, and automated fruit sorting, enabling efficient classification of fruits as unripe, ripe, or rotten based on their visual characteristics.

# Results

| Left-aligned | Center-aligned | Right-aligned | Left-aligned | Center-aligned | Right-aligned |
|     :---:    |     :---:      |     :---:     |     :---:    |     :---:      |     :---:     |
| git status   | git status     | git status    | git status   | git status     | git status    |
| git diff     | git diff       | git diff      | git diff     | git diff       | git diff      |
