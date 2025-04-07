# CNN models for ["Intel Image Classification"](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset.

## Project Overview
This project features two distinct Convolutional Neural Network (CNN) models:
1. A custom-built CNN trained from scratch.
2. A fine-tuned ResNet50 model leveraging using learning.

# Benchmarks
### 1. From scratch model:
   ![alt text](/assets/images/from_scratch_model_readme_graphs.png)

   ![alt text](/assets/images/from_scratch_model_readme_conf_matrix.png)

&nbsp;

### 2. Fine-tuned ResNet50 model:
   ![alt text](/assets/images/transfer_learning_model_readme_graphs.png)

   ![alt text](/assets/images/transfer_learning_model_readme_conf_matrix.png)

&nbsp;

# How to use
### 1. This repository includes pre-trained versions of both models. 
   - **`model_from_scratch.pth`** (Trained from scratch) 
   - **`model_transfer_learning.pth`** (Fine-tuned ResNet50)  

### 2. To use a model in your Python environment, place the desired .pth file in a directory of your choice and load it using the following code: 

   ```python
   import torch  

   model = torch.load('path/to/model_from_scratch.pth', weights_only=False)
   ```

### 3. To make predictions, pass a torch tensor of shape [num_images, channels, width, height] to the model. The output will be a tensor of shape [num_images, 6], where each value represents a class probability.

   ```python
   import torch  

   # Example input (batch of images)
   input_tensor = torch.randn(num_images, channels, width, height)  

   # Get model outputs
   outputs = model(input_tensor)  # Shape: [num_images, 6]  

   # Retrieve predicted class
   max_values, max_indices = torch.max(outputs, dim=1)  
   print("Predicted classes:", max_indices)  # Shape: [num_images]
   ```

### 4. To map the predicted numeric class values use the following code:
   ```python
   class_labels = {
      0: 'buildings',
      1: 'forest',
      2: 'glacier',
      3: 'mountain',
      4: 'sea',
      5: 'street'
   }

   predicted_labels = [class_labels[idx.item()] for idx in max_indices]
   print("Predicted labels:", predicted_labels)
   ```
