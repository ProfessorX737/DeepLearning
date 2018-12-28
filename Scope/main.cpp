#include <iostream>
#include "types.h"
#include "logging.h"
#include "Tensor.h"
#include "Variable.h"
#include "MatMul.h"
#include "Add.h"
#include "Initializer.h"
#include "Placeholder.h"
#include "Tanh.h"
#include "Square.h"
#include "Sub.h"
#include "Graph.h"
#include "Multiply.h"
#include "Optimizer.h"
#include <vector>

using namespace std;

//int main(void) {
//    Graph graph;
//    auto w1 = Variable(graph,{1,2},DT_FLOAT);
//    auto w2 = Variable(graph, {1,2}, DT_FLOAT);
//    auto sqrDiff = Square(graph, Sub(graph,w2,w1));
//    auto error = Multiply(graph, sqrDiff, 0.5f);
//    auto optimizer = OptimizerOp<float>(graph,error);
//    w1->init<float>({1,1});
//    w2->init<float>({3,3});
//    
//    Tensor dx;
//    optimizer.evalDeriv(dx);
//    cout << dx.matrix<float>() << endl;
//    cout << dx.dimString() << endl;
//	cin.get();
//    return 0;
//}

int main(void) {
	const int I = 2;
	const int H = 2;
	const int O = 1;
	const int BATCH_SIZE = 1;

	// get all our data and store into tensors
	Tensor data({ BATCH_SIZE,I }, DT_FLOAT);
	Tensor label({ BATCH_SIZE,1 }, DT_FLOAT);
	data.fill<float>({ 1,1 });
	label.fill<float>({ 1 });

	Graph graph;

	auto x = Placeholder(graph, { BATCH_SIZE,I });
	auto y = Placeholder(graph, { BATCH_SIZE,O });

	auto w1 = Variable(graph, { I,H }, DT_FLOAT);
	auto b1 = Variable(graph, { BATCH_SIZE,H }, DT_FLOAT);
	auto mult = MatMul(graph, x, w1);
	auto add = Add(graph, mult, b1);
	//auto h1 = Tanh(graph, add);
	auto w2 = Variable(graph, { H,O }, DT_FLOAT);
	auto b2 = Variable(graph, { BATCH_SIZE,O }, DT_FLOAT);
	auto h2 = Add(graph, MatMul(graph,add,w2), b2);
	auto sqrDiff = Square(graph, Sub(graph, h2, y));
	auto error = Multiply(graph, sqrDiff, 0.5f);

	//w1->init(RandomNormal<float>(0, 0.1));
	//w2->init(RandomNormal<float>(0, 0.1));
	w1->init<float>({ 2,2,2,2 });
	w2->init<float>({ 2,2 });
	b1->init(ZeroInit<float>());
	b2->init(ZeroInit<float>());


	//auto optimizer = OptimizerOp<float>(graph, error);
    auto optimizer = std::make_shared<OptimizerOp<float>>(graph,error);
	std::vector<Tensor> out;
    Graph::eval({ {x,data},{y,label} }, { optimizer }, out);
    
    for(int i = 0; i < gradients_.size(); i++) {
        cout << gradients_[i].matrix<float>() << endl << endl;
    }
    cout << out[0].matrix<float>() << endl;

	//cout << out[0].matrix<float>() << endl;
	//cout << out[1].matrix<float>() << endl;

	cin.get();
	return 0;
}
