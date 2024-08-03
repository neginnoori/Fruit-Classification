# Fruit Ripeness Classification
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
Sure, I'll continue the README file and include the information about the provided Jupyter notebook for training and testing. Here's the completed README for your project:

---
## Usage

### Prerequisites

To run the code in this repository, you need to have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

You can install the required packages using the following command:

```bash
pip install torch torchvision numpy matplotlib jupyter
```

### Dataset

Download the dataset from the FRINN GitHub repository and ensure it is organized in the following structure:

```
data/
    apple/
        ripe/
        unripe/
        rotten/
    banana/
        ripe/
        unripe/
        rotten/
    ...
```

### Training and Evaluation

The provided Jupyter notebook `Ripe_Unripe_Rotten_with_test.ipynb` contains the complete code for training and evaluating the fruit classification model. To get started, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/fruit-ripeness-classification.git
    cd fruit-ripeness-classification
    ```

2. Open the Jupyter notebook:
    ```bash
    jupyter notebook Ripe_Unripe_Rotten_with_test.ipynb
    ```

3. Follow the instructions in the notebook to load the dataset, train the model, and evaluate its performance.

### Fine-tuning and Customization

By utilizing this repository, you can explore the potential of CNNs and the VGG16 architecture for fruit recognition tasks, fine-tune hyperparameters, and adapt the code to your own fruit image dataset. The trained model can find applications in various domains such as agriculture, food quality assessment, and automated fruit sorting, enabling efficient classification of fruits as unripe, ripe, or rotten based on their visual characteristics.

## Contributing

Contributions are welcome! If you have any improvements or suggestions, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
