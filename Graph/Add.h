#pragma once

#include "BinaryOperator.h"
#include "Graph.h"

template<typename T>
class Add : public BinaryOperator<T> {
private:
	T binaryOperation(T left, T right) {
		return left + right;
	}
public:
	Add(Graph<Node<T>>& g, Node<T>& left, Node<T>& right) : BinaryOperator<T>(g,left,right) {}
	~Add() {}
};
