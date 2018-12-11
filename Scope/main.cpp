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

#include <vector>

using namespace std;

void main(void) {
	Graph graph;
	auto v1 = Variable(graph, { 2,2 }, DT_FLOAT);
	v1.init<float>({ 1,2,3,4 });
	auto v2 = Variable(graph,{ 2,2 }, DT_FLOAT);
	v2.init<float>({ 1,1,1,1 });
	Tensor in({ 2,2 }, DT_FLOAT);
	in.fill<float>({ 2,2,2,2 });

	auto p1 = Placeholder(graph, {2,2});

	auto add = Add<float>(graph, v1, v2);
	auto mult = MatMul<float>(graph, add, p1);
	auto mult2 = MatMul<float>(graph, add, v2);
	auto tanh = Tanh<float>(graph, mult2);

	std::vector<Tensor> out;
	graph.eval({ { p1.getId(),in } }, { &mult, &add, &tanh}, out);

	cout << out[0].matrix<float>() << endl;
	cout << out[1].matrix<float>() << endl;
	cout << out[2].matrix<float>() << endl;

	cin.get();
}