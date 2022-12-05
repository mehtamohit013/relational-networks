# Relational Networks for VQA

 Mohit Mehta(mm12318@nyu.edu)&nbsp;&nbsp;&nbsp; Umang Shah(uks8451@nyu.edu)&nbsp;&nbsp;&nbsp; Surya Narayan(sn3402@nyu.edu)
<hr>



# Dataset
The dataset used for VQA generally comprises images and questions that can be answered from the image. The question can broadly be classified into 
1. Relational : What is the shape of the object that is farthest from the gray object?  

2. Non-Relational : What is the shape of the gray object.

### Challenges: 
It is highly challenging to produce an accurate dataset capturing Visual QA relations due to the following facts:

1. Absence of rigourous and complete vocabularly
2. Require knowledge about the real world that cannot be captured in the training data
3. Contain Ambiguities and linguistic biases which are then carried into the model.

### Solution: CLEVR
To mitigate these issues, the CLEVR dataset was proposed which contains 3D-rendered objects such as spheres, cubes and cylinders having different (not distinct) colors, with different material types.

<!-- Check the Question Example Provided -->
In order to accurately answers these question, a model not only needs to understand object attributes such as location, shape, color and material, but also need to be accurately capture the relational data between two objects. Some example of relational questions include "Is the cube the same material as the cylinder?".

### Implementation
For this project, we are using Sort-of-CLEVR dataset which differ from CLEVR dataset in following ways:

- Image contain only 2D objects instead of 3D objects present in CLEVR
- Each Image contains exactly 6 objects, all of distinct color, with shape being a circle or a rectangle/square. 
- Sort-of-CLEVR dataset seperates relational question from non relational question with 10 relational and 10 non-relational questions for each image. However, the training has been done jointly on both relational and non-relational questions.


# Relational Networks:
In order to accurately capture the relations between objects for relational questions, relational networks has been used, which is given by: 
    $$ RN(O) = f_{\phi} (\Sigma_{i,j}g_{\theta}(o_i,o_j)) $$

where $O = \{ o_1, o_2, o_3, . . .o_n \}$ dentoes the object set, with $ f_{\phi} $ and $ g_{\theta} $ being MLPs with $\phi$ and $\theta$ as synaptic weights learned by the model.

Here, $g_{\theta}$ helps us in infering the result whether the two objects are related, which is then passed through $f_{\phi}$ which determine the best answer based on all the relations.

The above formulation of relation networks has following strengths:

- **They can infer relations without any prior knowledge**
    RNs does not depends on any prior knowledge of relation between the two objects. Thereofore, RNs must learn to infer the existence and implications of object relations

- **Data efficient** 
    RNs only use a single function for computing relations where the input is a object pair which not only generalizes relations but also batching operation resulting in increase in speed considerably. In contrast, in a traditional MLP based approach the model has to learn and embed $n^2$ (where n is the number of objects) identical functions within its weight parameters to account for all possible object pairings.. Therefore, the cost of learning a relation function $n^2$ times using a single feedforward pass per sample, as in an MLP, is replaced by the cost of $n^2$ feedforward passes per object set(i.e., for each possible object pair in the set) and learning a relation function just once, as in an RN.

- **Order invariant**
    The summation in RN equation not only makes the RNs invariant to the order of objects in the input, but also ensures that the relation is generally representative of the relation.

## Model

The paper discusses the following models, which differ in the input provided to them
### 1.  CNN + LSTM + RN 

Since inherently RN cannot deal with pixels, CNN is used to infer a set of objects from the input images and LSTM is used to infer the questions.

Futhermore, to capture the relation between the question and object pair, RNs has been modified to also taken question embedding, which has been produced by passing question through a LSTM, to accurately capture the relations

$$ RN(O,q) = f_{\phi} (\Sigma_{i,j}g_{\theta}(o_i,o_j,q)) $$

However, for the current implementation on Sort-of-CLEVR dataset, as the question length is same, the question has been converted into embeddings, eliminating the need of LSTMs.

As for CNN, the input images are of size 128 x 128 which are convolved through 4 layers to generate <I>k</I> feature map of size <I>d x d</I>.The current implementation uses <I>k = 24</I> and <I>d = 3</I> with <I>stride = 2</I>  and <I>padding = 1</I> used at each level. Batch Normalization is also used here to ensure all features are processed equally disregarding the scale of values. This also acts as a way of regularizing thus eliminating the need for dropouts. This also makes the model less variant to hyperparameter (learning rate).ReLU was used as non-linearity applied after batch-norm in this implementation.

**Snippet of CNN**: 
```Python
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

<!-- It is important to know that the positions of each object are arbitrarily assigned by the RN and not something learned by the CNN. -->

### 2. State Description + LSTM + RN
For this task, we replace the CNN by the state description data present in Sort-of-CLEVR dataset. The data contains the location, shape, size and color of all the objects present in a img. The following objects is then passed to RN accompanied by the Question Embeddings produced by LSTMs.

### 3. Other models
The above theory has also been tested on multiple dataset such as bAbI dataset. For bAbI, where information about the object also needs to be extracted from text, the paper suggests identifying about 20 sentences as support sets which were immediately before the probe question. These sentences were also tagged with the position they occurred relative to the question and processed word by word using LSTM. 

## Implementations

### Task 1: CNN + RN
- Here we take image + question embedding as input.
- The CNN implementation to extract object information from images is as discussed above. 


- Creating pairs from object set. 
    Each object of size 25 x 26 is mapped to every other object of the same shape. And all such object pairs from all images in batch along with the question embeddings are eventually converted into a tensor of shape 40000 x 70 which are then passed to the relation operation.

- Computing relation $g_{\theta}(o_i,o_j)$ 
In our implementation, the relation part of our RNN consists of 4 fully connected neural layers having the dimensions as discussed below
    - Layer 1:<br/>
    Input: (batch_size * (24+2) * 2+18 ) x 70
    where 24 is the number of kernels, 2 is the size coordinate representation for the object. This is multiplied by 2 since we process pair of objects at a go. 18 is the size of the embedding for the question.
    Output: 256

    - Layer 2 - 4:<br/>
    Input = Output = batch_size x 256

        ```Python
        # initialized in model
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

    Summation is done over each batch for all object pairs resulting in an object if size batch_size x 256

    ``` Python
    # part of training process
    if self.relation_type == 'ternary':
            x_g = x_.view(mb, (d * d) * (d * d) * (d * d), 256)
        else:
            x_g = x_.view(mb, (d * d) * (d * d), 256)

        x_g = x_g.sum(1).squeeze()
    ```

- Calculting $f_{\phi}$ <br/>
Single layer with input = output = 256
    ```Python
    self.f_fc1 = nn.Linear(256, 256)
    ```
- Fully Connected Output Layers: <br>
    A fully connected network of 2 layers is used to get the final output from the model

    - Layer1: (Input: 64 x 256, Output: 64 x 256)
    - Dropout Layer with dropout probablity = 0.5
    - Layer2: (Input: 64 x 256, Output: 64 x 10)
    - Log SoftMax : (Output: 64 x 10)

    ```Python
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
- Optimizer: <br/>
    The CNN + RN is trained using Adam Gradient Descent Optimizer.

- Tuning of model:\
    When setting the seed value to 42 with a learning rate = 0.0001,  an accuracy of 97% on training data and 91%  on test data was achieved for relational questions after 40 epochs.\
    Similarly, an accuracy of 99% was achieved on both test and training data for the non-relational model

    Non relational accuracy over epoch.
    ![Non-Relational Accuracy](images/task-1-unary-accuracy.png)

    Relational accuracy over epoch
    ![Non-Relational Accuracy](images/task1-binary-accuracy.png)

    [CSV for relational train accuracy](csvs/run-Dec02_04-41-09_1fb17b539b50_Accuracy_train_unary-tag-Accuracy_train.csv)

    [CSV for relational test accuracy](csvs/run-Dec02_04-41-09_1fb17b539b50_Accuracy_test_unary-tag-Accuracy_test.csv)

    [CSV for non-relational train accuracy](csvs/run-Dec02_04-41-09_1fb17b539b50_Accuracy_train_binary-tag-Accuracy_train.csv)

    [CSV for non-relational test accuracy](csvs/run-Dec02_04-41-09_1fb17b539b50_Accuracy_test_binary-tag-Accuracy_test.csv)

- Other attempts at tuning\
    When training the model with a higher learning rate (0.0005), It took 60 epochs for the accuracies to stabilize to (3% in the test and 96% on training data. \
    Since we achieve a good enough accuracy with low epochs, we consider aforementioned result as a more practical outcome.

### Task 2: State Description + LTSM + RN
In this task, we used state description as an input rather than an image. This eliminates the need for CNN as this information can directly be processed by RNN.

The training proces has been futher explained below:
- For Sort-of-CLEVR dataset, the state description for each image contains 6 objects with the following properties: ```[center_x, center_y, shape, color, size]```. All text values are replaced by integers starting from 1 by creating appropriate vocabularies.

- For generating Question embeddings, we used LSTM with hidden layer size of 128.

One of the benifits of using state description model is that the model completely avoids the complexities and errors arising due to CNN and focus on relation part better.


#### LSTM

- As per the paper, LSTM was implemented having the following configurations.
    - Questions are passed to LSTM with shape ```[batch_size x w x v]```, where w is the number of tokens in question and v is the size of vocab (39)
    - The hidden state is of size 128.
    - We used randomly generated weights for initial state ```h0``` ( the hidden state at the zero timestamp), ```c0``` (the cell state at the zero timestamp). They can be alternatively initialized to zero
    - We then pass the final output vector from LSTM to RN as question embedding with shape being ```[batch_size,128]```

#### RN 
- RN in this case only differs from task 1 in terms of input. 

- The input layer will take a vector of shape
```[(batch_size * 6 * 6),4]```

- Subsequent hidden layers will stay of same dimensions as before (task 1)

- The output layer is of dimension ```[batch_size,vocab_size]```.

#### Tuning

We acheive the best test result by using a learning rate of <b>0.0003</b> resulting in training and testing accuracy of around <b>94%</b> after <b>40</b> epochs.


![Non-Relational Accuracy](images/task2-combined-accuracy.png)


# Results
Our implementation reproduces almost the same state-of-the-art accuracy stated by the paper [A simple neural network module
for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf). We acheived an relational accuracy 91% when using CNN + LSTM + RN and 98% accuracy when using State Description + LSTM + RN. 

# Conculsion
This repository implements and test the relational networks introduced in the paper [A simple neural network module
for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf) for CNN + RN + LSTM architecture and State Description + RN + LSTM.

# Acknowledgement
The following work has been built upon the pytorch implementation of relational networks ([link](https://github.com/kimhc6028/relational-networks)) by [kimhc6028](https://github.com/kimhc6028)