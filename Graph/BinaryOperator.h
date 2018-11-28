#pragma once

#include "Node.h"
#include "Graph.h"

template<typename T>
class BinaryOperator : public Node<T> {
protected:
	Node<T>* m_left;
	Node<T>* m_right;
	virtual T binaryOperation(T left, T right) = 0;
public:
	BinaryOperator(Graph<Node<T>>& g, Node<T>& left, Node<T>& right) : Node<T>(g) {
		m_left = &left;
		m_right = &right;
	}
	T evaluate() {
		return binaryOperation(m_left->evaluate(), m_right->evaluate());
	}
	void collectInputs(std::vector<Node<T>*>& inputs) {
		m_left->collectInputs(inputs);
		m_right->collectInputs(inputs);
	}
};
