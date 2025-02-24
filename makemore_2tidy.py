import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

'''
build dataset
'''
#load dataset and create character mappings
words = open("names.txt","r").read().splitlines()
chars = sorted(list(set("".join(words))))
stringtoindex = {ch:i+1 for i,ch in enumerate(chars)}
stringtoindex["."] = 0
indextostring = {i+1:ch for i,ch in enumerate(chars)}
indextostring[0] = "."

block_size = 3#context

def build_dataset(words):
    X ,Y = [], []
    for w in words:
        context = [0] * block_size#for each new word, start with [0,0,0] - > '.','.','.'
        for ch in w + ".":
            index = stringtoindex[ch]
            X.append(context)# contains our current context X = [0,0,0]
            Y.append(index)# contains the target which is 5 for "e" (emma)
            context = context[1:] + [index] # now we slide the window so that our current context is [0,0,5]

    #convert to tensors
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X,Y
    
def data_split(words):
    train_words,temp_words = train_test_split(words,test_size=0.2,random_state=42)
    val_words,test_words = train_test_split(temp_words,test_size=0.5,random_state=42)

    X_train , Y_train = build_dataset(train_words)
    X_val, Y_val = build_dataset(val_words)
    X_test, Y_test = build_dataset(test_words)

    return X_train, Y_train,X_val,Y_val,X_test,Y_test

def initialize_parameters(block_size,embedding_dimension,activation):
    #blocksize = context
    #layer_1 = num of neurons in first layer
    num_layers = int(input("State number of layers (including output layer and excluding input layer):"))
    layer_sizes = [block_size*embedding_dimension]
    for i in range(num_layers):
        num_neurons = int(input(f"State number of neurons of layer {i+1}:"))
        layer_sizes.append(num_neurons)

    parameters = {}
    for i in range(num_layers):#starting from first hidden layer
        parameters[f'W{i+1}'] = torch.empty(layer_sizes[i+1],layer_sizes[i]) * 0.01
        #initialzing weights based on activation chosen in the network
        if activation == "tanh":
            nn.init.xavier_normal_(parameters[f'W{i+1}'])#ensures variance stability (vanishing gradients as gradients might shrink too small)
        if activation == "relu":
            nn.init.kaiming_normal_(parameters[f'W{i+1}'], mode='fan_in', nonlinearity='relu')#prevents exploding gradientsis the 
        parameters[f'W{i+1}'].requires_grad = True
        parameters[f'b{i+1}'] = torch.zeros(layer_sizes[i+1])
        parameters[f'b{i+1}'].requires_grad = True
    parameters["embedding"] = torch.randn(27,embedding_dimension) * 0.1
    parameters["embedding"].requires_grad=True

    return parameters

def get_layer_sizes(parameters):
    layer_sizes = []
    i= 1
    while True:
        key = f"W{i}"
        if key in parameters:
            layer_sizes.append(parameters[key].shape[0])
            i = i +1
        else:
            break
    return layer_sizes

def forward(x,parameters,activation): 
    
    embedding = parameters["embedding"]
    emb = embedding[x]
    #print(f"embedding size:{emb.shape}")
    emb = emb.view(emb.shape[0],-1)
    #print(f"embedding size:{emb.shape}")
    #we get num of neurons of each layer by checking the shape[0] of weights of each layer
    #layer1 has W1.shape=[100,6] and W2.shape = [27,100]
    layer_sizes = get_layer_sizes(parameters)#note that here layer_sizes[0] corresponds to layer1
    #now we need to extract num of layers
    num_layers = sum(1 for key in parameters if key.startswith('W'))#w0w python cool 
    #first activation is our embedding layer as input 
    #forward pass
    h = emb
    for i in range(1,num_layers+1):

        
        W = parameters[f"W{i}"]
        b = parameters[f"b{i}"]

        h = h @ W.T + b #linear transformation

        if i < num_layers:#checking to see if final layer or not 
            if activation == "tanh":
                h = torch.tanh(h)
            if activation == "relu":
                h = torch.relu(h)
        


    logits = h
    return logits


def backward(y,logits,lr,parameters):

    loss = F.cross_entropy(logits,y)

    for param in parameters.values():
            if param.grad is not None:
                param.grad.zero_()#zero the gradients so old ones dont mix with new ones 

    loss.backward()


    with torch.no_grad():
        for param in parameters.values():
                if param.grad is not None:
                    param.data = param.data - lr*param.grad

    return loss.item()

def train_model(x_train,y_train,x_val,y_val,x_test,y_test,parameters,activation):
    batch_size = int(input("Enter desired batch size:"))
    num_epochs = int(input("Enter desired epoch size:"))
    lr= float(input("Enter desired learning rate:"))
    


    for epoch in range(1,num_epochs+1):
        permuatation = torch.randperm(x_train.shape[0])
        x_train = x_train[permuatation]
        y_train = y_train[permuatation]
        loss_train = 0

        total_train_loss = 0 
        total_train_correct = 0
        total_samples = 0
        for i in range(0,x_train.shape[0],batch_size):#iterating thru all batches

            x_batch_train = x_train[i:i+batch_size]
            y_batch_train = y_train[i:i+batch_size]

            logits = forward(x_batch_train,parameters,activation)

            loss_train = backward(y_batch_train,logits,lr,parameters)
            
            total_train_loss += loss_train * y_batch_train.size(0)#why???????

            predictions_train = torch.argmax(logits, dim=1)
            total_train_correct += (predictions_train == y_batch_train).sum().item()
            total_samples += y_batch_train.size(0)
        
        avg_loss = total_train_loss / total_samples
        accuracy = total_train_correct / total_samples

        with torch.no_grad():#freezing the parameters
            val_logits = forward(x_val, parameters,activation)
            val_loss = F.cross_entropy(val_logits, y_val)
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = (val_preds == y_val).float().mean()

        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {avg_loss:.4f} | Train Acc: {accuracy*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")


    with torch.no_grad():
        test_logits = forward(x_test, parameters,activation)
        test_loss = F.cross_entropy(test_logits, y_test)
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = (test_preds == y_test).float().mean()
        print(f"\nFinal Test Performance: | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    return parameters
    



def sampling(parameters,block_size,activation):
    #try adding temperature
    #input = context
    context = [0] * block_size
    x = torch.tensor([context])
    name = ""
    while True:
        logits = forward(x,parameters,activation)
        probs = F.softmax(logits,dim=1)
        index = torch.multinomial(probs, num_samples=1).item()
        context = x.tolist()[0][1:]+ [index]
        x= torch.tensor([context])
        character = indextostring[index]
        name+=character
        if index == 0:
            break
            
    return name

    
            

print("---------------")
print("---------------")
print("---------------")
print("INITIALIZATIONS:")
print("---------------")
X,Y = build_dataset(words)
print(f"Shape of X: {X.shape}\nShape of Y: {Y.shape}")

x_train, y_train,x_val,y_val,x_test,y_test = data_split(words)
print(f"x_train shape:{x_train.shape}\ny_train shape:{y_train.shape}\nx_val shape:{x_val.shape}\ny_val shape:{y_val.shape}\nx_test shape:{x_test.shape}\ny_test shape:{y_test.shape}")

activation = (input("Enter desired activation function (relu or tanh):"))

parameters = initialize_parameters(3,10,activation)
for key, value in parameters.items():
    if isinstance(value, torch.Tensor):  # For weights & biases
        print(f"{key} shape: {value.shape}")
    elif isinstance(value, nn.Embedding):  # For embedding layer
        print(f"{key} shape: {value.weight.shape}")  # Access weight matrix of embedding
print("---------------")
print("---------------")
print("---------------")
print("MODEL TRAINING:")
print("---------------")


final_parameters = train_model(x_train,y_train,x_val,y_val,x_test,y_test,parameters,activation)
print("---------------")
print("---------------")
print("---------------")
print("SAMPLING")
num_samples = int(input('State number of names you would like to sample:'))
for i in range(num_samples):
    name = sampling(final_parameters,3,activation)
    print(f"Name {i+1}:{name}\n")


print("---------------")
print("---------------")
print("---------------")

'''
hp i used:
tanh
3 layers,128,64,27
batch size = 128
epochs = 64
lr = 0.1
'''