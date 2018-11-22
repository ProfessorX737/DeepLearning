#include <iostream>
#include <vector>
#include "Multiply.h"
#include "Add.h"
#include "Variable.h"
#include "Graph.h"
#include "Node.h"

using namespace std;

int main(int argc, char ** argv) {
	
	Graph<Node<int>> g;
	auto five = Variable<int>(g,5);
	//auto three = Variable<int>(g,3);
	//auto add1 = Add<int>(g, five, three);
	//auto two = Variable<int>(g, 2);
	//auto mult1 = Multiply<int>(g, two, add1);
	cout << five.evaluate();
	int temp;
	cin >> temp;
	return 0;
}