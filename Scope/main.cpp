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
#include <vector>
#include "GradientDescent.h"
#include "MomentumDescent.h"
#include "Sigmoid.h"

using namespace std;

//int main(void) {
//    Graph graph;
//    auto w1 = Variable(graph,{1,2},DT_FLOAT);
//    //auto w2 = Variable(graph, {1,2}, DT_FLOAT);
//    auto sqr = Square(graph, w1);
//    //auto error = Multiply(graph, sqrDiff, 0.5f);
//	auto tanh = Tanh(graph, sqr);
//    auto optimizer = std::make_shared<OptimizerOp<float>>(graph,tanh);
//    w1->init<float>({0.5f,0.6f});
//
//	//std::vector<Tensor> out;
//	Tensor out;
//	Graph::eval(optimizer, out);
//    for(int i = 0; i < gradients_.size(); i++) {
//        cout << gradients_[i].matrix<float>() << endl << endl;
//    }
//    
//    //Tensor dx;
//    //optimizer.evalDeriv(dx);
//    //cout << dx.matrix<float>() << endl;
//    //cout << dx.dimString() << endl;
//	cin.get();
//    return 0;
//}

//int main(void) {
//	const int I = 2;
//	const int H = 4;
//	const int O = 1;
//	const int BATCH_SIZE = 1;
//
//	// get all our data and store into tensors
//	Tensor data({ BATCH_SIZE,I }, DT_FLOAT);
//	Tensor label({ BATCH_SIZE,1 }, DT_FLOAT);
//	data.fill<float>({ 1,0 });
//	label.fill<float>({ 1 });
//
//	Graph graph;
//
//	auto x = Placeholder(graph, { BATCH_SIZE,I });
//	auto y = Placeholder(graph, { BATCH_SIZE,O });
//
//	auto w1 = Variable(graph, { I,H }, DT_FLOAT);
//	auto b1 = Variable(graph, { BATCH_SIZE,H }, DT_FLOAT);
//	auto mult = MatMul(graph, x, w1);
//	auto add = Add(graph, mult, b1);
//	auto h1 = Tanh(graph, add);
//	auto w2 = Variable(graph, { H,O }, DT_FLOAT);
//	auto b2 = Variable(graph, { BATCH_SIZE,O }, DT_FLOAT);
//	auto h2 = Add(graph, MatMul(graph,h1,w2), b2);
//	auto sqrDiff = Square(graph, Sub(graph, y, h2));
//	auto error = Multiply(graph, sqrDiff, 0.5f);
//
//	w1->init(RandomNormal<float>(0, 0.1));
//	w2->init(RandomNormal<float>(0, 0.1));
////	w1->init<float>({ 2,2,2,2 });
////	w2->init<float>({ 2,2 });
//	b1->init(ZeroInit<float>());
//	b2->init(ZeroInit<float>());
//
//    auto optimizer = std::make_shared<GradientDescentOp<float>>(graph,error,0.2f);
//    
//    std::srand(static_cast<uint>(std::time(NULL)));
//
//    for(int i = 0; i < 10; i++) {
//        const int d1 = rand() % 2;
//        const int d2 = rand() % 2;
//        const int d3 = d1 ^ d2;
//        data.asVec<float>().data()[0] = d1;
//        data.asVec<float>().data()[1] = d2;
//        label.asVec<float>().data()[0] = d3;
//        std::vector<Tensor> out;
//        Graph::eval({ {x,data},{y,label} }, { h2, optimizer }, out);
//        cout << "===========================" << endl;
//        cout << d1 << " " << d2 << ": " << d3 << endl;
//        cout << "output: " << out[0].asVec<float>() << endl;
//        cout << "error: " << out[1].asVec<float>() << endl << endl;
//        std::vector<Tensor> gradients = optimizer->getGradients();
//        for(int i = 0; i < gradients.size(); i++) {
//            cout << gradients[i].matrix<float>() << endl << endl;
//        }
//        cout << "===========================" << endl;
//    }

    

    //cout << w1->tensor().matrix<float>() << endl;

//	cin.get();
//	return 0;
//}

int main(void) {
	const int I = 2;
	const int H = 2;
	const int O = 1;
	const int BATCH_SIZE = 1;

	// get all our data and store into tensors
	Tensor data({ BATCH_SIZE,I }, DT_DOUBLE);
	Tensor label({ BATCH_SIZE,O }, DT_DOUBLE);
	data.fill<double>({ 1,1 });
	label.fill<double>({ 0 });

	Graph graph;

	auto x = Placeholder(graph, { BATCH_SIZE,I }, DT_DOUBLE);
	auto y = Placeholder(graph, { BATCH_SIZE,O }, DT_DOUBLE);

	auto w1 = Variable(graph, { I,H }, DT_DOUBLE);
	auto b1 = Variable(graph, { BATCH_SIZE,H }, DT_DOUBLE);
	auto mult = MatMul(graph, x, w1);
	auto add = Add(graph, mult, b1);
	auto h1 = Tanh(graph, add);
	auto w2 = Variable(graph, { H,O }, DT_DOUBLE);
	auto b2 = Variable(graph, { BATCH_SIZE,O }, DT_DOUBLE);
	auto h2 = Tanh(graph,Add(graph, MatMul(graph,h1,w2), b2));
	auto error = Square(graph, Sub(graph, y, h2));
	//auto error = Multiply(graph, sqrDiff, 0.5);

	w1->init(RandomNormal<double>(0, 0.5));
	w2->init(RandomNormal<double>(0, 0.5));
//	w1->init<double>({ 0.15,0.25,0.2,0.3 });
//	w2->init<double>({ 0.4,0.5 });
	b1->init(RandomNormal<double>(0,0.5));
	b2->init(RandomNormal<double>(0,0.5));
//  b1->init<double>({0.35,0.35});
//  b2->init<double>({0.6});

    auto optimizer = std::make_shared<MomentumDescentOp<double>>(graph,error,0.2f,0.8f);
    
    std::srand(static_cast<uint>(std::time(NULL)));

    for(int i = 0; i < 2000; i++) {
        const int d1 = rand() % 2;
        const int d2 = rand() % 2;
        const int d3 = d1 ^ d2;
        data.asVec<double>().data()[0] = static_cast<double>(d1);
        data.asVec<double>().data()[1] = static_cast<double>(d2);
        label.asVec<double>().data()[0] = static_cast<double>(d3);
        std::vector<Tensor> out;
        Graph::eval({ {x,data},{y,label} }, { h2,optimizer }, out);
        cout << d1 << " " << d2 << ": " << d3 << endl;
        cout << "output: " << out[0].asVec<double>() << endl;
        cout << "error: " << out[1].asVec<double>() << endl << endl;
    }

//        std::vector<Tensor> gradients = optimizer->getGradients();
//        for(int i = 0; i < gradients.size(); i++) {
//            cout << gradients[i].matrix<float>() << endl << endl;
//        }

	cin.get();
	return 0;
}