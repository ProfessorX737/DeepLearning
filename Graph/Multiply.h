#pragma once

#include "BinaryOperator.h"
#include "Node.h"
#include "Graph.h"

template<typename T>
class Multiply : public BinaryOperator<T> {
private:
	T binaryOperation(T left, T right) {
		return left * right;
	}
public:
	Multiply(Graph<Node<T>>& g, Node<T>& left, Node<T>& right) : BinaryOperator<T>(g,left,right) {}
	~Multiply() {}
};