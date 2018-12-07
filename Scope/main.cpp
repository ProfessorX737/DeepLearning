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
	auto v1 = Variable({ 2,2,2 },DT_FLOAT);
	v1.init<float>({ 1,2,3,4,5,6,7,8 });
	auto v2 = Variable({ 2,2,2 }, DT_FLOAT);
	v2.init<float>({ 1,1,1,1,1,1,1,1 });

	auto m = Add<float>(v2, v2);
	Tensor out; 
	m.eval(out);

	cout << out.tensor<float,3>() << endl;

	int in;
	cin >> in;
}