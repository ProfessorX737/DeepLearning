#include <iostream>
#include <Eigen/Dense>
#include <random>
#include "RandomNormal.h"
#include "FullyConnected.h"
#include "ZeroInitializer.h"
#include "BinaryStep.h"
#include "TanhActivation.h"
#include <Eigen/CXX11/Tensor>

using namespace std;
using namespace Eigen;

float activate(float val) {
	return (val > 0) ? 1.0 : 0.0;
}

int main(int argc, char ** argv) {
	RandomNormal<> weight_init(0.0, 0.1);
	ZeroInitializer<> bias_init;
	BinaryStep<> step;
	TanhActivation<> tanh;
	FullyConnected<4,4> layer(&weight_init, &bias_init, &tanh);
	cout << layer.getWeights() << endl << endl;
	layer.applyActivation();
	cout << layer.getWeights() << endl << endl;
	cout << layer.getBiases() << endl << endl;
	MatrixXf input(1, 4);
	input << 1, 2, 3, 4;
	cout << layer.feed(input) << endl;
	//cout << "chicken";
	int a;
	cin >> a;

	FullyConnected<2,2> layer1(&weight_init, &bias_init, &tanh);
	FullyConnected<1,1> layer2(&weight_init, &bias_init, &tanh);
	
	return 0;
}