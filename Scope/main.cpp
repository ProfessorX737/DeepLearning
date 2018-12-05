#include <iostream>
#include "types.h"
#include "logging.h"
#include "Tensor.h"

#include <vector>

using namespace std;

void main(void) {
	std::vector<Tensor::Index> dims;
	dims.push_back(4);
	dims.push_back(1);
	Tensor t({ 4,1 });
	t.init({ 1.0f,2.0f,3.0f,4.0f});
	cout << t.shaped<float, 2>({4,1}) << endl;
	cout << t.shaped<float, 2>({1,4}) << endl;

	cout << t.dimString() << endl;

	int in;
	cin >> in;
}