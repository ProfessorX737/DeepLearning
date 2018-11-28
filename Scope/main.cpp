#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include "types.h"


using namespace std;

void main(void) {
	EnumToDataType<DT_FLOAT>::type afloat;
	afloat = 1.6f;
	cout << afloat;
	int in;
	cin >> in;
}