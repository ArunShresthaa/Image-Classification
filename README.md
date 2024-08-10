```markdown

# Deep Learning Model Training in PyTorch



This repository contains the code for training a deep learning model for bird classification using PyTorch. The model is a customized ResNet architecture trained on an image dataset. The code includes data augmentation, learning rate scheduling, and early stopping mechanisms.



## Dependencies



The following Python libraries are required:



- `torch`

- `torchvision`

- `numpy`

- `tqdm`

- `matplotlib`



## Hyperparameters



The key hyperparameters used in the script are:



```python

num_classes = 25

batch_size = 64

num_epochs = 50

initial_lr = 0.001

weight_decay = 1e-4

dropout_rate = 0.5

patience = 10  # for early stopping

```



## Device Configuration



The script is configured to use GPU if available, otherwise, it falls back to the CPU.



```python

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

```



## Model Definition



The model is a variant of ResNet customized for this task. The `BasicBlock` and `CustomResNet` classes define the model architecture.





```python

class BasicBlock(nn.Module):

    expansion = 1



    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)



        self.downsample = None

        if stride != 1 or in_planes != planes:

            self.downsample = nn.Sequential(

                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(planes)

            )



    def forward(self, x):

        identity = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)



        if self.downsample:

            identity = self.downsample(x)



        out += identity

        out = self.relu(out)

        return out



class CustomResNet(nn.Module):

    def __init__(self, block, layers, num_classes=25):

        super(CustomResNet, self).__init__()

        self.in_planes = 64



        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(

            nn.Dropout(p=0.5),

            nn.Linear(512 * block.expansion, num_classes)

        )



    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []

        layers.append(block(self.in_planes, planes, stride))

        self.in_planes = planes

        for _ in range(1, blocks):

            layers.append(block(planes, planes))

        return nn.Sequential(*layers)



    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)



        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)



        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)



        return x



def custom_resnet18(num_classes=25):

    return CustomResNet(BasicBlock, [2, 2, 2, 2], num_classes)

```



## Data Loading and Transformation



Data augmentation and normalization are applied to the training and validation datasets.



```python

data_transforms = {

    'train': transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.RandomRotation(30),

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.4731, 0.4819, 0.4018], std=[0.1925, 0.1915, 0.1963])

    ]),

    'val': transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.4706, 0.4802, 0.4020], std=[0.1907, 0.1898, 0.1950])

    ]),

}



data_dir = './Seen Datasets'

image_datasets = {x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x])

                  for x in ['train', 'val']}

data_loaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size,

                                   shuffle=True, num_workers=8, pin_memory=True)

                for x in ['train', 'val']}



dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

```



## Model Training



### Setup



The loss function, optimizer, and learning rate scheduler are defined before starting the training process.



```python

model = custom_resnet18(num_classes=num_classes).to(device)



criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=initial_lr, 

                                          steps_per_epoch=len(data_loaders['train']), epochs=num_epochs)

```



### Training Loop



The training loop iterates over the epochs, performing training and validation in each epoch. Early stopping is implemented to avoid overfitting.



```python

train_loss_history = []

train_acc_history = []

val_loss_history = []

val_acc_history = []



best_model_wts = None

best_acc = 0.0

early_stop_counter = 0



for epoch in range(num_epochs):

    print(f'Epoch {epoch}/{num_epochs - 1}')

    print('-' * 10)



    for phase in ['train', 'val']:

        if phase == 'train':

            model.train()

        else:

            model.eval()



        running_loss = 0.0

        running_corrects = 0



        for inputs, labels in tqdm(data_loaders[phase], desc=f"{phase} - Epoch {epoch+1}"):

            inputs = inputs.to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)



            optimizer.zero_grad()



            with torch.set_grad_enabled(phase == 'train'):

                outputs = model(inputs)

                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)



                if phase == 'train':

                    loss.backward()

                    optimizer.step()



            running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds == labels.data)



        if phase == 'train':

            scheduler.step()

            epoch_train_loss = running_loss / dataset_sizes['train']

            epoch_train_corrects = running_corrects.double() / dataset_sizes['train']

        else:

            epoch_val_loss = running_loss / dataset_sizes['val']

            epoch_val_corrects = running_corrects.double() / dataset_sizes['val']



    train_loss_history.append(epoch_train_loss)

    train_acc_history.append(epoch_train_corrects.item())

    val_loss_history.append(epoch_val_loss)

    val_acc_history.append(epoch_val_corrects.item())



    print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_corrects:.4f}')

    print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_corrects:.4f}')



    if epoch_val_corrects > best_acc:

        best_acc = epoch_val_corrects

        best_model_wts = model.state_dict().copy()

        early_stop_counter = 0

    else:

        early_stop_counter += 1



    if early_stop_counter >= patience:

        print("Early stopping")

        model.load_state_dict(best_model_wts)

        break



print('Best val Acc: {:4f}'.format(best_acc))



if best_model_wts:

    model.load_state_dict(best_model_wts)



torch.save(model, 'best_bird_classifier_full.pth')

print("Model saved with architecture and weights")

```



## Plotting the Training Curves



The script plots the training and validation loss and accuracy curves for visualization.



```python

epochs = len(train_loss_history)



plt.figure(figsize=(12, 5))



plt.subplot(1, 2, 1)

plt.plot(range(epochs), train_loss_history, label='Train Loss')

plt.plot(range(epochs), val_loss_history, label='Val Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Training and Validation Loss')

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(range(epochs), train_acc_history, label='Train Accuracy')

plt.plot(range(epochs), val_acc_history, label='Val Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Training and Validation Accuracy')

plt.legend()



plt.show()

```



This markdown document provides a structured description of your training code. You can further customize it by adding more sections or details as needed.

```