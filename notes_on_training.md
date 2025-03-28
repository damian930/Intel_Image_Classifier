All the tessting of different parameters, layers... go here 
    No need to have it all sit in the same file

```python
Model 1.1: 
    Regular torch cnn totoial weights
        (
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool  = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1   = nn.Linear(16 * 5 * 5, 120)
            self.fc2   = nn.Linear(120, 84)
            self.fc3   = nn.Linear(84, 10)
        )
    Epochs        -> 30
    Optimizer     -> Adam
    Leraning rate -> 0.001
    Weight decay  -> 0.00001
    Result: 
```
![alt text](/assets/images/model_1-1_graphs.png)

<hr style="width: 100%; height: 3px; background-color: yellow;">

```python
Model 1.2: 
    (
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
    )
    Epochs        -> 30
    Optimizer     -> Adam
    Leraning rate -> 0.001
    Learning rate mid training changes:
        if(epoch % 5 == 0 and epoch != 0):
            for group in optimizer.param_groups:
                group['lr'] /= 10
    Weight decay  -> 0.00001
    Result: 
```
![alt text](/assets/images/model_1-2_graphs.png)

<hr style="width: 100%; height: 3px; background-color: yellow;">

```python
Model 1.3: 
    (
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
    )
    Epochs        -> 25
    Optimizer     -> Adam
    Leraning rate -> 0.0001
    Weight decay  -> 0.00001
    Result: 
```
![alt text](/assets/images/model_1-3_graphs.png)

<hr style="width: 100%; height: 3px; background-color: yellow;">

```python
Model 2.1: 
    (
        self.conv1  = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2  = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3  = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1    = nn.Linear(in_features=64*18*18, out_features=500)
        self.fc2    = nn.Linear(in_features=500, out_features=50)
        self.fc3    = nn.Linear(in_features=50, out_features=len(labels_numbers_to_strings))
    )
    Epochs        -> 25
    Optimizer     -> Adam
    Leraning rate -> 0.001
    Weight decay  -> 0.00001
    Result: 
```
![alt text](/assets/images/model_2-1_graphs.png)

<hr style="width: 100%; height: 3px; background-color: yellow;">

```python
Model 2.2: 
    (
        self.conv1  = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2  = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3  = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1    = nn.Linear(in_features=64*18*18, out_features=500)
        self.fc2    = nn.Linear(in_features=500, out_features=50)
        self.fc3    = nn.Linear(in_features=50, out_features=len(labels_numbers_to_strings))
    )
    Epochs        -> 25
    Optimizer     -> Adam
    Leraning rate -> 0.01
    Weight decay  -> 0.00001
    Result: 
```
![alt text](/assets/images/model_2-2_graphs.png)
        
<hr style="width: 100%; height: 3px; background-color: yellow;">

```python
Model 2.3: 
    (
        self.conv1  = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2  = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3  = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1    = nn.Linear(in_features=64*18*18, out_features=500)
        self.fc2    = nn.Linear(in_features=500, out_features=50)
        self.fc3    = nn.Linear(in_features=50, out_features=len(labels_numbers_to_strings))
    )
    Epochs        -> 25
    Optimizer     -> Adam
    Leraning rate -> 0.0001
    Weight decay  -> 0.00001
    Result: 
```
![alt text](/assets/images/model_2-3_graphs.png)

<hr style="width: 100%; height: 3px; background-color: yellow;">

```python
Model 2.4 (Current Best Version): 
    (
        self.conv1    = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2    = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3    = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1      = nn.Linear(in_features=64*18*18, out_features=500)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2      = nn.Linear(in_features=500, out_features=50)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3      = nn.Linear(in_features=50, out_features=len(labels_numbers_to_strings))
    )
    Epochs        -> (<= 25)
    Optimizer     -> Adam
    Leraning rate -> 0.001
    Weight decay  -> 0.00001
    Result: 
```
![alt text](/assets/images/model_2-4_graphs.png)