#pragma once

#include "Node.h"
#include "Graph.h"

template<typename T>
class UnaryOperator : public Node<T> {
protected:
	Node<T>* m_child;
	virtual T unaryOperation(T var) = 0;
public:
	UnaryOperator(Graph<Node<T>>& g, Node<T>& child) : Node<T>(g) {
		m_child = &child;
	}
	T evaluate() {
		return unaryOperation(m_node->evaluate());
	}
	void collectInputs(std::vector<Node<T>*>& inputs) {
		m_child->collectInputs(inputs);
	}
};
