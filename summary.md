# Relational Networks for VQA

## Dataset
Dataset used for VQA generally comprises of images and questions that can be answered from the image. The question can broadly be classified into 
1. relational : 
What is the shape of object that is fartest from the gray object? or non-relational  
2. non relational : 
What is the shape of gray object.

### Difficulties: 
Most Visual QA datasets have following issues which make them training for VQA models difficult.

1. Full vocabulary is not known
2. Require knowledge about the realworld that cannot be captured in the training data
3. Contain Ambiguities and linguistic biases which are then carried into the model.
### Solution: CLEVR
To avoid these issues, CLEVR dataset was proposed which contains 3D-rendered objects such as spheres, cubes and cylinders having different (not distinct) colours.

Queries are formed that require understanding attributes such as location, shape, color and material.

e.g: Is the cube same material as the cylinder? What is the color of sphere? 

### Implementation Used: Sort-of-clevr

Dataset genereted randomly that differs on following aspects from CLEVR datset

1. Seperates relational and non-relational questions.
2. Images contains 2D objects
3. Each image has 6 objects of randomly chosen shape (square or circle)
4. Each object in an image will have distinct color from the set of 6 colors (red, blue, green, orange, yellow, gray)
5. Questions are hard-coded as fixed-length binary strings to avoid NLP related complexities and errors
6. Around 10 relational and 10 non-relational question for each image.


## Relational Networks: 
Relational Networks are neural networks which can be mathematically described by below given equaion.
    $$ RN(O) = f_{\phi} (\Sigma_{i,j}g_{\theta}(o_i,o_j)) $$
    where
    $$O = \{ o_1, o_2, o_3, . . .o_n \}$$
    $ f_{\phi} $ and $ g_{\theta} $ are functions which will be MLPs with $\phi$ and $\theta$ being synaptic weights learned by the model

Output of $g_{\theta}$ infers if the two object passed are related and in what way, thus aptly called 
a relation.

Relational Networks have three notable strengths:
1. They infer relations <br/>
- Since they operate on all pairs, they  do not need to know beforehand which objects are related. 
2. Data efficient (Also makes them good candidate to be used in one shot learnings )<br/>
- RNs use only one function for computing relations where input is object-object pair thus can better generalise relations.
- In contrast if a traditional MLP approch were all $n$ objects are passed at once as input, it would have to learn and embed $n^2$ functions resulting in a high cost of learning for $n^2$ feedforward passes 

2. Order invariant ( Since they operate on a set of objects)<br/>
- This is ensured by the summation in the functional form.
- Also ensures that output of RN contains information of relations that generally exist in the object set.

## Models

The paper discusses three models which differ by the input provided to them
### 1. Pixels 

Since inherently RN cannot deal with pixels, CNN is used to infer set of objects from input. 

Input images are of size 128 x 128 which are convolved through 4 layers to generate k feature map of size d x d. 

In current implementation k = 24 and d = 3
with stride = 2  and padding = 1 used at each level. Batch Normalization is also used here to ensure all features are processed equally disregarding the scale of values. This also acts as a way of relgularizing thus eliminating the need for dropouts. This also makes model less delicate to hyperparameter (learning rate). 

ReLu was used as non-linearity applied after batchnorm in this implementation.

snippet: 
```
class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x
```
The output is of size : batch_size x k x (d + 2 x padding) x (d + 2 x padding)

It is important to know that positions of each object is arbitrarily assigned by the RN and not something learned by the CNN.

### 2. State Description

since the RN does not take pixels of image as input anyway, Input can be described by specifing shape, color, x_coordinate, y_coordinate, etc of each object in object-object pair as a vector. 

### 3. Question

The type of relations learned by RN should be question dependent. For the implementation modifies the RN architecture that can be functionally represented as below. where q denotes the question embedings. 

$$ RN(O) = f_{\phi} (\Sigma_{i,j}g_{\theta}(o_i,o_j,q)) $$


The question embeddings can be generated using LSTM. A vocalbulary should be created so that each word from question can be assigned a unique integer enabling the LSTM to generate the question embedding.

At each timestep 1 word from sentence is passed as input to LSTM. LSTM here makes sense due to their ability to memorize important stuff. By propagating the final state of LSTM to RN we ensure this memory is passed to RN. 

For the current implementation, Sort-of-ClLEVR saves these embeding eliminating the need for LSTM. These are used to get the desired accuracy with CNN+RN model. For state description model LSTM was used anyway.

### 4. Natural Language

For dataset such as bAbI, where information about object also needs to be extracted from text, the paper suggests to identify about 20 sentences as support set which were immediately prior to the probe question. These sentences were also tagged with the position they occured relative to the question and processed word by word using LSTM. 

## Implementations

### Task 1: CNN + RN
- Here we take image + question embedding as input.
- The CNN implementation to extract object information from image is as discussed above. 

- Creating pairs from object set. <br/>
    Each object of size 25 x 26 is mapped to every other object of same shape. And al such opject pairs from all images in batch along with the question embeddings are eventually converted into a tensor of shape 40000 x 70 which are then passed to the relation opertaion.

- relation $g_{\theta}$ <br/>
In our implementation, relation part of our RNN consists of 4 layer neural network the dimensions of which are as follows
    - Layer 1: <br/>
    Input: (batch_size * (24+2) * 2+18 ) x 70
    where 24 is the number of kernels, 2 is the size co-ordinate representation for object. This is multiplied by 2 since we process pair of objects at a go. 18 is the size of embedding for question.<br/>
    Output: 256

    - Layer 2 - 4: <br/>
    Input = Output = batch_size x 256
    ```# initialized in model
    if self.relation_type == 'ternary':
            ##(number of filters per object+coordinate of object)*3+question vector
            self.g_fc1 = nn.Linear((24+2)*3+18, 256)
        else:
            ##(number of filters per object+coordinate of object)*2+question vector
            self.g_fc1 = nn.Linear((24+2)*2+18, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        
    ```
- Summation of $g_{\theta}$ over set $O$

    Summation is done over each batch for all obect pairs resulting in an object if size batch_size x 256

    ``` # part of training process
    if self.relation_type == 'ternary':
            x_g = x_.view(mb, (d * d) * (d * d) * (d * d), 256)
        else:
            x_g = x_.view(mb, (d * d) * (d * d), 256)

        x_g = x_g.sum(1).squeeze()
    ```

- $f_{\phi}$ <br/>
Single layer with input = output = 256
    ```
    self.f_fc1 = nn.Linear(256, 256)
    ```
- Output Layer: <br>
    A fully connected netowrk of 2 layers is used to get the final output from the model

    - Layer1: <br/>
    input: 64 x 256<br/>
    output: 64 x 256

    - dropout: 0.5

    - Layer2: <br/>
    input: 64 x 256<br/>
    output: 64 x 10<br/>

    - Log Soft Max (Converting Regression to Classification)
    output: 64 x 10

    ```
    class FCOutputModel(nn.Module):
        def __init__(self):
            super(FCOutputModel, self).__init__()

            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 10)

        def forward(self, x):
            x = self.fc2(x)
            x = F.relu(x)
            x = F.dropout(x)
            x = self.fc3(x)
            ans = F.log_softmax(x, dim=1)
            return ans
    ```
### Task 2: State Description + LTSM + RN

- In this task we take state description as input instead of image. This eliminates the need for CNN as this information can directly be processed by RNN

- Also, instead of using question embedding directly, we use LSTM which outputs a vector of size 128 for each question.

- LSTM <br/>
todo: explain

- RN <br/>
todo: explain dimensions of each layers



