#include <iostream>
#include <vector>
#include "Multiply.h"
#include "Add.h"
#include "Variable.h"
#include "Graph.h"
#include "Node.h"
#include "Input.h"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char ** argv) {

	VectorXf m1(2);

	m1 << 1, 2;
	Matrix<float, 2, 2> m2;
	m2 << 1, 2,
		3, 4;
	MatrixXf m3 = m1.transpose() * m2;
	cout << m3 << endl;
	std::cin.get();
	
	//Matrix2f mat2;
	//mat2 << 1, 2, 3, 4;
	//Graph<Node<Matrix2f>> g;
	//auto five = Variable<Matrix2f>(g, mat2);
	//auto x = Input<Matrix2f>(g, mat2);
	//cout << x.getId() << endl;
	//auto add1 = Add<Matrix2f>(g, five, x);
	//Matrix<float, 1, 2> mat12;
	//mat12 << 1, 2;
	//auto two = Variable<Matrix<float,1,2>>(g, mat12);
	//auto mult1 = Multiply<Matrix2f>(g, two, add1);
	//cout << mult1.evaluate() << endl;
	//std::vector<Node<Matrix2f>*> inputs;
	//g.getInputs(mult1,inputs);
	//cout << "inputs: ";
	//for (auto i : inputs) {
	//	cout << i->getId() << ", ";
	//}
	////cout << add3.evaluate();
	//int temp;
	//cin >> temp;
	return 0;
}