#include <iostream>
#include <vector>
#include <ctime>
#include <tuple>

#include "Graph.h"
#include "Variable.h"

//#include "logging.h"
#include "MatMul.h"
#include "Add.h"
#include "Initializer.h"
#include "Placeholder.h"
#include "Tanh.h"
#include "Square.h"
#include "Sub.h"
#include "Multiply.h"

#include "GradientDescent.h"
#include "MomentumDescent.h"
#include "Sigmoid.h"
#include "Reduce.h"
#include <time.h>


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
//	const int H = 2;
//	const int O = 1;
//	const int BATCH_SIZE = 1;
//
//	// get all our data and store into tensors
//	Tensor data({ BATCH_SIZE,I }, DT_FLOAT);
//	Tensor label({ BATCH_SIZE,O }, DT_FLOAT);
//	data.fill<float>({ 1,1 });
//	label.fill<float>({ 0 });
//
//	Graph graph;
//
//	auto x = Placeholder(graph, { BATCH_SIZE,I }, DT_FLOAT);
//	auto y = Placeholder(graph, { BATCH_SIZE,O }, DT_FLOAT);
//
//	auto w1 = Variable(graph, { I,H }, DT_FLOAT);
//	auto b1 = Variable(graph, { 1,H }, DT_FLOAT);
//	auto h1 = Tanh(graph,Add(graph, MatMul(graph,x,w1), b1));
//	auto w2 = Variable(graph, { H,O }, DT_FLOAT);
//	auto b2 = Variable(graph, { 1,O }, DT_FLOAT);
//	auto h2 = Tanh(graph,Add(graph, MatMul(graph,h1,w2), b2));
//	auto error = Square(graph, Sub(graph, y, h2));
//    auto reduce = ReduceMean<0>(graph,error);
//
//	w1->init(RandomNormal<float>(0, 0.5));
//	w2->init(RandomNormal<float>(0, 0.5));
////	w1->init<float>({ 0.15,0.25,0.2,0.3 });
////	w2->init<float>({ 0.4,0.5 });
//	b1->init(RandomNormal<float>(0,0.5));
//	b2->init(RandomNormal<float>(0,0.5));
////    b1->init<float>({0.35,0.35});
////    b2->init<float>({0.6});
//    
//    std::vector<Tensor> out;
//    Graph::eval({ {x,data},{y,label} }, { error }, out);
//    cout << out[0].asVec<float>() << endl;
//
//    auto optimizer = MomentumDescent(graph, reduce, 0.25f, 0.5f);
//
//    std::srand(static_cast<uint>(std::time(NULL)));
//
//    for(int i = 0; i < 2000; i++) {
//        const int d1 = rand() % 2;
//        const int d2 = rand() % 2;
//        const int d3 = d1 ^ d2;
//        data.asVec<float>().data()[0] = static_cast<float>(d1);
//        data.asVec<float>().data()[1] = static_cast<float>(d2);
//        label.asVec<float>().data()[0] = static_cast<float>(d3);
//        std::vector<Tensor> out;
//        Graph::eval({ {x,data},{y,label} }, { h2, optimizer }, out);
//        cout << "===========================" << endl;
//        cout << d1 << " " << d2 << ": " << d3 << endl;
//        cout << "output: " << out[0].asVec<float>() << endl;
//        cout << "error: " << out[1].asVec<float>() << endl << endl;
//
//    }
//
//
////    cout << w1->tensor().matrix<float>() << endl;
//
//	cin.get();
//	return 0;
//}

int main(void) {
	const int I = 2;
	const int H = 2;
	const int O = 1;
	const int BATCH_SIZE = 2;

	// get all our data and store into tensors
	Tensor data({ BATCH_SIZE,I }, DT_DOUBLE);
	Tensor label({ BATCH_SIZE,O }, DT_DOUBLE);
//	data.fill<double>({ 0,1,
//                        1,1 });
//	label.fill<double>({ 1,
//                         0 });
//	data.fill<double>({ 1,1 });
//	label.fill<double>({ 0 });

	Graph graph;

	auto x = Placeholder(graph, { BATCH_SIZE,I }, DT_DOUBLE);
	auto y = Placeholder(graph, { BATCH_SIZE,O }, DT_DOUBLE);

	auto w1 = Variable(graph, { I,H }, DT_DOUBLE);
	auto b1 = Variable(graph, { 1,H }, DT_DOUBLE);
	auto h1 = Tanh(graph,Add(graph, MatMul(graph,x,w1), b1));
	auto w2 = Variable(graph, { H,O }, DT_DOUBLE);
	auto b2 = Variable(graph, { 1,O }, DT_DOUBLE);
	auto h2 = Tanh(graph,Add(graph, MatMul(graph,h1,w2), b2));
	auto error = Square(graph, Sub(graph, y, h2));

	w1->init(RandomNormal<double>(0, 0.6));
	w2->init(RandomNormal<double>(0, 0.6));
//	w1->init<double>({ 0.15,0.25,0.2,0.3 });
//	w2->init<double>({ 0.4,0.5 });
	b1->init(RandomNormal<double>(0,0.6));
	b2->init(RandomNormal<double>(0,0.6));
//    b1->init<double>({0.35,0.35});
//    b2->init<double>({0.6});
    
//    std::vector<Tensor> out;
//    Graph::eval({ {x,data},{y,label} }, { error }, out);
//    cout << out[0].asVec<double>() << endl;

    //auto optimizer = std::make_shared<MomentumDescentOp<double>>(graph,error,0.25f,0.5f);
    auto optimizer = MomentumDescent(graph, error, 0.2f, 0.5f);
   // auto optimizer = GradientDescent(graph, error, 0.35f);
    auto reduce = ReduceMean<0>(graph,optimizer);
    //auto sgd = GradientDescent(graph, error, 0.5f);

    //std::srand(static_cast<uint>(std::time(NULL)));


    clock_t tStart = clock();
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
        
        std::vector<Tensor> out;
        Graph::eval({ {x,data},{y,label} }, { reduce }, out);
//        cout << "output: " << out[0].asVec<double>() << endl;
//        cout << "error: " << out[1].asVec<double>() << endl << endl;
        cout << "error: " << out[0].asVec<double>() << endl;
    }
    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

#if defined(_WIN32) || defined(WIN32)
	cin.get();
#endif
	return 0;
}

//int main(void) {
//    Tensor t({2,3},DT_FLOAT);
//    t.fill<float>({1,2,3,4,5,6});
//    Tensor t2;
//    t2.sharedCopyInit<float>(t, 0);
//    cout << t2.tensorPadRight<float, Tensor::MAX_DIMS>() << endl;
////    TTypes<float,2>::Tensor tensor(2,1);
////    typename TTypes<float,Tensor::MAX_DIMS>::Tensor newT = Tensor::tensorPadRight<float, 2,Tensor::MAX_DIMS>(tensor);
////    Eigen::array<int, 1> reduceIndices = {0};
////    cout << t.tensorPadRight<float, Tensor::MAX_DIMS>().mean(reduceIndices) << endl;
////    cout << tensor << endl;
//    return 0;
//}