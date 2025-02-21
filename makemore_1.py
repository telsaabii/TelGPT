import torch
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.nn.functional as F
words = open("names.txt","r").read().splitlines()
namescount = len(words)
minword = min([len(word) for word in words])
maxword = max([len(word) for word in words])

#bigramlanguage model works with 2 characters at a time. the character we are at (context) and the next character (target)



#visualizing the counts of bigrams through dictionary
'''
counts = {} # counting frequency of letters
for w in words:#just the first word
    chs = ["<start>"] + list(w) + ["<end>"] # turning word into list
    for ch1,ch2 in zip(chs,chs[1:]):# pairing emma with mma -> (e,m) (m,m) (m,a) (a,m)
        bigram = (ch1,ch2)#key
        if bigram in counts:
            counts[bigram] += 1
        else:
            counts[bigram] = 1
    
    
for bigram,count in sorted(counts.items(),key = lambda x: x[1],reverse = True):
    print(f"{bigram},{count}") 
'''

N = torch.zeros((27,27),dtype = torch.int32)#this is our bigram frequency table 

#preparing the vocabulary (letters)
chars = set(''.join(words))#set of all unique characters
chars = list(chars)#converting to list
chars = sorted(chars)#sorting
stringtoindex = {ch:i+1 for i,ch in enumerate(chars)}#dictionary that maps characters to indices
stringtoindex["."] = 0#representing the start or end of a word
indextostring = {i+1:ch for i,ch in enumerate(chars)}#so we can index index to character
indextostring[0] = "."
print(f"chars: {chars}")
print(f"string to index: {stringtoindex}")
print(f"index to string: {indextostring}")

for w in words:
    characters = ["."] + list(w) +["."] # adding start and end tokens to the words
    for ch1,ch2 in zip(characters,characters[1:]):#geting the bigrams
        #print(f"character1: {ch1}")
        #print(f"character2: {ch2}")
        index1 = stringtoindex[ch1]#index of first character
        index2 = stringtoindex[ch2]#index of second character
        #print(f"index1: {index1}")
        #print(f"index2: {index2}")
        N[index1,index2] += 1 #incrementing the count of the bigram in the tensor 

#tarek
#arek
#ta ar re ek
#visualizing the bigram frequency table

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = indextostring[i] + indextostring[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
plt.show()
print("-----------------")
print("Bigram model approach")
print("-----------------")
#each row represents the probability distribution of the next character given the current character
#now we need to get probabilities from the counts and sample from the distribution
#g = torch.Generator().manual_seed(214783647)
P = N.float()#convert the whole tensor to float
#model smoothing by adding 1 to every bigram count like so this is done so that we dont get any zero probabilities -> negative nll
P = (N+1).float()
#now we're gonna need to make each row a probability distribution
P /= P.sum(dim=1,keepdim = True) #NORMALIZING EACH ROW OF P TO A SUM OF 1 

ix = 0 #signifies "."
names = []
for i in range(10): 
    while True:
        p = P[ix] #retrieves row of probabilities corresponding to character ix 
        #p = N[ix].float()
        #p = p/p.sum() #getting row probabilities
        ix = torch.multinomial(p,1,replacement=True,generator = None).item()#sampling from the distribution
        #print(indextostring[ix])
        names.append(indextostring[ix])
        if ix == 0:
            break 

print(f"Generated names: {''.join(names)}")



#P[i,j] is the probability of character j given character i

log_likelihood = 0
n = 0
for w in words : 
    characters = ["."] + list(w) + ["."]
    for ch1,ch2 in zip(characters,characters[1:]):
        index1 = stringtoindex[ch1]
        index2 = stringtoindex[ch2]
        prob = P[index1,index2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n+=1
        #print(f"p({ch2}|{ch1}) = {prob}")

nll = -log_likelihood
nll = nll/n
print(f"Negative Log Likelihood (average): {nll}")# need to minimize

#neural network approach
print("-----------------")
print("Neural Network Approach")
print("-----------------")
#our input is a character in the sequence
#our output is a probability distribution over the characters in the vocabulary
import torch.nn as nn 
import torch.nn.functional as F
 

xs,ys = [],[]#input and output sequences
for w in words:
    characters = ["."] + list(w) + ["."]
    for ch1,ch2 in zip(characters,characters[1:]):
        xs.append(stringtoindex[ch1])
        ys.append(stringtoindex[ch2])

xs = torch.tensor(xs)
print(f"xs: {xs}")
ys = torch.tensor(ys)
print(f"ys: {ys}")
num = xs.nelement()
#forwardpass!!!
#doesnt make sense to directly input the character index into the model so we need to one hot encode the input
x_encoded = F.one_hot(xs,num_classes = 27).float()
print(f"x_encoded shape = {x_encoded.shape}")
g = generator = torch.Generator().manual_seed(2147483647)
weights = torch.randn((27,27),requires_grad=True)# N neurons - > (27,N)
print(f"weights shape: {weights.shape}")
#activation = x_encoded @ weights 

#print(f"activation = {activation}")

#our neural net will be a single layer 
#so we get (N,27) as inputs such that N is the number of letters in the word and our output will be (27,27) since we're taking 27 letters as input and outputting the probability distribution of all the letters 
#lets use softmax as our activation function

#optimization : minimize loss by tuning weights - compute gradients of loss wrt to weights
#we're doing classification so we will use negative log likelihood 
 
'''
xs = [0,5,13,13,1] corresponds to . e m m a 
ys = [5,13,13,1,0] corresponds to e m m a .
so when input is 0(".") then output probability is probs[0,5]
when input is 5("e") then output probability is probs[1,13]
and so on 
so our focus will be on optimizing probs[0,5],probs[1,13],probs[2,13],probs[3,1],probs[4,0]
'''


weights.grad = None#w.grad tells us the influence of the each weight on the  loss functio
for i in range(100):#gradient descent
    #forward pass
    x_encoded = F.one_hot(xs,num_classes=27).float()
    logits = x_encoded @ weights
    probs = F.softmax(logits,dim = 1)
    regularization_term = (weights**2).mean()
    loss = -probs[torch.arange(num),ys].log().mean() + 0.01 * regularization_term
    weights.grad = None
    loss.backward()
    #now we update weights
    #with torch.no_grad():
        #weights -= 0.01 * weights.grad
    weights.data += -50 * weights.grad
    if i % 10 == 0:
        print(f"loss after optimization {i}: {loss}")

print("-----------------")
print("Sampling From neural net")
print("-----------------")
#now we have our weights ready so we can "sample" from the neural net
print
output = []

for i in range(5):
    ix = 0 #always start with "."
    while True:#break when we reach "."
        x_encoded = F.one_hot(torch.tensor([ix]),num_classes=27).float()
        logits = x_encoded @ weights
        probs = F.softmax(logits,dim=1)
        ix = torch.multinomial(probs,num_samples=1,replacement=True).item()#.item extracts number
        value = indextostring[ix]
        #print(value)
        output.append(value)
        if ix == 0:
            break

print(''.join(output))