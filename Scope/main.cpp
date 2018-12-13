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

#include <vector>

using namespace std;

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
	auto h1 = Tanh(graph,Add(graph, MatMul(graph,x,w1), b1));
	auto w2 = Variable(graph, { H,O }, DT_FLOAT);
	auto b2 = Variable(graph, { BATCH_SIZE,O }, DT_FLOAT);
	auto h2 = Add(graph, MatMul(graph,h1,w2), b2);
	auto diff = Sub(graph, h2, y);
	auto sqrDiff = Square(graph, diff);

	//w1.init(RandomNormal<float>(0, 0.1));
	//w2.init(RandomNormal<float>(0, 0.1));
	w1->init<float>({ 2,2,2,2 });
	w2->init<float>({ 2,2 });
	b1->init(ZeroInit<float>());
	b2->init(ZeroInit<float>());

	std::vector<Tensor> out;
	graph.eval({ {x,data},{y,label} }, { sqrDiff }, out);

	cout << out[0].matrix<float>() << endl;

	cin.get();
	return 0;
}
