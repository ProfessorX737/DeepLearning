#include <iostream>
#include "types.h"
#include "logging.h"
#include "Tensor.h"
#include "Variable.h"

#include <vector>

using namespace std;

void main(void) {
	//Tensor t({ 2,2,2 });
	//t.init<float>({ 1,2,3,4,
	//				5,6,7,8 });
	//float* data = t.data<float>();
	//for (int i = 0; i < t.numElements(); i++) {
	//	cout << data[i] << ",";
	//}
	Variable v({ 3 },DT_FLOAT);
	v.fill<float>({ 1,2,3 });


	cout << v.t_.asVec<float>() << endl;

	int in;
	cin >> in;
}