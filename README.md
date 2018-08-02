#  LTOM(Layer Training by Output Mirror) 100 times faster Neural Network model compare with traditional backpropagation NN

The neural network becomes very popular nowadays and is used to solve complex AI and ML problems. However, the neural network is very expensive to train due to the traditional complex backpropagation techniqueue. 
Surprisingly, our new model uses very simple method to train the NN without any complex backpropagation method and achieve almost similar successful rate. 
In this model, we train each layer starting from first layer using input as output of pervious layer and output as mirror of training set output.

1. For example, first layer has n1 neurons and output(Y) has ny dimension then convert ny dimension  output(Y) into n1 dimension vector:
 Y:ny-->Y:n1
 Here is simple algorithm forconverting output(Y) into n1 dimension vector:

```python
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
   ``` 

    
2. Then, calculate weight w  for this layer from input(X) and mapped output(Ym)
  W = X dot Inverse(Ym) 
3. Calculate bias b
  b= f(W.X)- Ym
4. Calculate output for this layer
   Yi  = f(W.X + b)
5. Move to next layer assign input X = Yi( output of previous layer) then repeat steps - 1-5 until  end

Here is complete algorithm  the training : 

```python
def train(X, Y, layout, activations):
    xi = X
    W=[]
    B=[]
    for i in range(len(layout)):
        if i < len(layout)-1  :
            yi = mappedToSize(Y, layout[i]) 
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
```

The prediction algorithm is very simple:
```python
def eval(X, W, B, layout, activations):
    xi = X
    for i in range(len(layout)):
        wi=W[i]
        bi=B[i]
        xf = xi
        if i < len(layout)-1:
            xi =activations[i](np.dot((xi), wi)+bi)
        else :
            xi =softmax(np.dot((xi), wi)+bi)
    return xi
```


 Training time for backpropagation network : 10.833237171173096 Seconds
 Training time for this network  : 0.17676091194152832 Seconds
 
 Both are using same dimension of layers and same activation function.
 
 Prediction report for the BP Model :
 
    precision    recall  f1-score   support

          0       0.85      1.00      0.92        34
          1       0.77      0.98      0.86        47
          2       1.00      0.88      0.94        43
          3       0.98      0.98      0.98        41
          4       1.00      0.89      0.94        47
          5       1.00      0.98      0.99        45
          6       1.00      0.89      0.94        55
          7       0.93      1.00      0.96        54
          8       0.97      0.78      0.86        40
          9       0.91      0.95      0.93        44

avg / total       0.94      0.93      0.93       450

Prediction report for our Model :

      precision    recall  f1-score   support

          0       1.00      1.00      1.00         9
          1       0.88      1.00      0.93         7
          2       1.00      1.00      1.00        10
          3       1.00      0.88      0.93         8
          4       1.00      1.00      1.00         8
          5       0.89      1.00      0.94         8
          6       1.00      1.00      1.00        13
          7       1.00      1.00      1.00        10
          8       1.00      0.80      0.89        10
          9       0.88      1.00      0.93         7

avg / total       0.97      0.97      0.97        90

The full code is available in the github.

