#pragma once

#include "Node.h"
#include "Graph.h"

template<typename T> 
class Variable : public Node<T> {
private:
	T m_val;
public:
	Variable(Graph<Node<T>>& g, T val) : Node<T>(g), m_val(val) {}
	T evaluate() {
		return m_val;
	}
};
