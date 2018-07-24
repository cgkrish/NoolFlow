#  TDIOM(Training using Direct Input & Output Data Model) 100 times faster Neural Network model compare with traditonal backpropagation NN

The neural network is very popular nowdays and used to solve complex AI and ML problems. However, the neural network is very expensive to train due to the traditional complex backpropagation techniqueue. 
Surprisingly, our new model uses very simple method to train the NN without any complex backpropagation method and achieve almost similar sucessful rate. 
In this model, we train each layer starting from first layer using input as output of pervious layer and output as mirror of training set output.
1. For example, first layer has n1 neurons and output(Y) has ny dimension then convert ny dimension  output(Y) into n1 dimension vector:
 Y:ny-->Y:n1
 Here is simple algotham to convert  from output(Y) into n1 dimension vector:

def mappedToSize(inputs, newSize):
    outputs = np.zeros((inputs.shape[0], newSize) )
    m = inputs.shape[1]
    n= newSize
    w = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            outputs[:,i] += (n -np.abs(i-j)-1)*inputs[:,j]*2/(m*(2*n -m-1)) 
            w[i,j]=(n -np.abs(i-j)-1)
    w=normalize(w.T)
    #outputs = np.dot( inputs, w)
    return (outputs)
    
2. Then, calculate weight w  for this layer from input(X) and mapped output(Ym)
  W = X dot Inverse(Ym) 
3. Calculate bias b
  b= f(W.X)- Ym
4. Calculate output for this layer
   Yi  = f(W.X + b)
5. Move to next layer assign input X = Yi( output of previous layer) then repeat steps - 1-5 until  end

Here is complete algorithm  the training : 

def train(X, Y, layout, activation):
    xi = X
    W=[]
    B=[]
    for i in range(len(layout)):
        if i < len(layout)-1  :
            yi = expandToSize(Y, layout[i]) 
        else:
            yi = Y
        xf = xi    
        wi = (np.dot(np.linalg.pinv((xi)), yi))
        bi = np.mean(activations[i](np.dot((xi), wi)) -yi, axis=0)
        
        if i < len(layout)-1:
            xi =activations[i]((np.dot((xi), wi)+bi))           
        else :
            xi =softmax(np.dot((xi), wi)+bi)         
        W.append(wi)
        B.append(bi)
    return W, B
