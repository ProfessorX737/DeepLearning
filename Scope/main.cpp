#include <iostream>
#include "types.h"
#include "logging.h"
#include "Tensor.h"
#include "Variable.h"
#include "MatMul.h"
#include "Add.h"
#include "Initializer.h"

#include <vector>

using namespace std;


void main(void) {
	Graph graph;
	auto v1 = Variable(graph,{ 2,2 },DT_FLOAT);
	v1.init<float>({ 1,2,3,4 });
	auto v2 = Variable(graph,{ 2,2 }, DT_FLOAT);
	v2.init<float>({ 1,1,1,1 });

	auto add = Add<float>(graph,v1, v2);
	auto mult = MatMul<float>(graph, add, v2);
	auto mult2 = MatMul<float>(graph, add, v1);

	std::unordered_map<int, int> feed_map;

	std::vector<Tensor> out;


	//cout << out.matrix<float>() << endl;

	//cout << map[v2.getId()] << endl;


	int in;
	cin >> in;
}