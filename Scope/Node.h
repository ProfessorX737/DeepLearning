#pragma once
#include <string>
#include <vector>

template<typename T>
class Node {
public:
	Node() : {}
	virtual T evaluate() = 0;
private:
	string class_name_;
};
