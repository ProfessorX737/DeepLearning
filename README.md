# DeepLearning
### A basic neural network library and auto-differentiator for educational purposes

Example usage:
Suppose we want to train a simple xor neural network that has two inputs (I = 2), two hidden nodes (H = 2), one output (O = 1).
It will look something like this:
![Image of Yaktocat](https://i.stack.imgur.com/hDsUW.png)
First, create tensors that will store the training data
```c++
Tensor data({ BATCH_SIZE, 2 }, DT_DOUBLE);
Tensor label({ BATCH_SIZE, 1 }, DT_DOUBLE);
```
XOR example data

| data 1 | data 2 | label |
| ------- | ------- | ----- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Intatiate our graph like so
```c++
Graph graph;
```
The graph is where we will build up a representation of the neural network
Create placeholder objects that will hold the training data. Placeholders are objects holding data that change every iteration.
```c++
auto x = Placeholder(graph, { BATCH_SIZE, 2 }, DT_DOUBLE);
auto y = Placeholder(graph, { BATCH_SIZE, 1 }, DT_DOUBLE);
```
Now we define the rest of the graph. 
Each w_i represent the weights between the previous set of nodes and the next set of nodes.
```c++
auto w1 = Variable(graph, { I,H }, DT_DOUBLE);
```
The h_i objects represent the values evaluated up until the ith hidden layer. The b_i objects are the biases that help the neural network learn
```c++
auto b1 = Variable(graph, { 1,H }, DT_DOUBLE);
auto h1 = Tanh(graph,Add(graph, MatMul(graph,x,w1), b1));
auto w2 = Variable(graph, { H,O }, DT_DOUBLE);
auto b2 = Variable(graph, { 1,O }, DT_DOUBLE);
auto h2 = Tanh(graph,Add(graph, MatMul(graph,h1,w2), b2));
```
Create the error object that represents the error between the actual output value and the target label value
```c++
auto error = Square(graph, Sub(graph, y, h2));
```
Next we initialize the variables w_i, b_i using random normal distribution
```c++
w1->init(RandomNormal<double>(0, 0.6));
w2->init(RandomNormal<double>(0, 0.6));
b1->init(RandomNormal<double>(0,0.6));
b2->init(RandomNormal<double>(0,0.6));
```
Next, we define the object optimizer that represent the operation to update the weights of the neural network. In this example we are using the momentun decent strategy
```c++
auto optimizer = MomentumDescent(graph, error, 0.3f, 0.5f);
auto reduce = ReduceMean<0>(graph,optimizer);
```
The reduce operation outputs an error value which is calculated by averaging all the errors in the batch of training items

Here we are cheating a little bit by generating training data on the fly using the `^` (xor) operator
```c++
for(int i = 0; i < 2000; i++) {
    for(int i = 0; i < BATCH_SIZE; i++) {
        const int d1 = rand() % 2;
        const int d2 = rand() % 2;
        const int d3 = d1 ^ d2;
        data.asVec<double>().data()[i*2] = static_cast<double>(d1);
        data.asVec<double>().data()[i*2+1] = static_cast<double>(d2);
        label.asVec<double>().data()[i] = static_cast<double>(d3);
        //cout << d1 << " " << d2 << ": " << d3 << endl;
    }
    // we update the graph weights after every batch
    std::vector<Tensor> out;
    Graph::eval({ {x,data},{y,label} }, { reduce }, out);
    cout << "error: " << out[0].asVec<double>() << endl;
}
```
