# README for ImageGPT Linear Probing Project

## How to Run Experiments

1. Clone or download the project repository.
2. Ensure that Python and the necessary libraries (`transformers`, `datasets`, `torch`, `numpy`, `matplotlib`, `sklearn`, etc.) are installed. You can install them using pip: `pip install transformers datasets torch numpy matplotlib sklearn`.
3. Run the Jupyter Notebook provided (`Final-Project.ipynb`) in a Jupyter environment (like Jupyter Lab or Google Colab).
4. The notebook is self-contained and will guide you through the process of loading datasets, preprocessing, model training, evaluation, and visualization.
5. Follow the instructions in each cell of the notebook to execute the code segments sequentially.

## Code Files Description

### a) Code Files Copied from Other Repositories

- The general instructions and code structure for using `ImageGPTFeatureExtractor` are borrowed from the following sources:
  - Hugging Face Transformers Documentation: [ImageGPT](https://huggingface.co/docs/transformers/model_doc/imagegpt)
  - GitHub Repository: [GenerativePretraining-from-Pixels](https://github.com/Jainam2410/GenerativePretraining-from-Pixels/blob/main/iGPT-CIFAR.ipynb)

### b) Modified Code Files

- The code segments for loading the pretrained ImageGPT model and feature extraction from the ImageGPT model are adapted from the above sources. Modifications include adjustments for batch processing, feature extraction specifics, and integration with the rest of the project pipeline.

### c) Original Student Code

- The remaining parts of the code, including training logistic regression, k-Nearest Neighbors, and Multi-Layer Perceptron classifiers, as well as the application of t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction, are the student's original work.

## Dataset Description

- The project utilizes the CIFAR-10 dataset, a widely used dataset in machine learning for object recognition. It consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.
- The dataset is accessed and loaded using the `datasets` library from Hugging Face. For training and testing purposes, a subset of 5,000 images from both training and test splits is used.
- The dataset is integrated into the project via the following code segment:

```python
from datasets import load_dataset
# Load a subset of the CIFAR-10 dataset
dataset = load_dataset('cifar10', split={'train': 'train[:5000]', 'test': 'test[:5000]'})
```

This README file provides a comprehensive guide to run the experiments, understand the codebase structure, and clarifies the origin and modifications of the code used in the project.
