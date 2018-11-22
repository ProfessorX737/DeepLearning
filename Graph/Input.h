#pragma once

#include "Node.h"

template<typename T> 
class Input : public Node<T> {
private:
	T m_val;
public:
	Input(Graph<Node<T>>& g, T val) : Node<T>(g), m_val(val) {
		g.addInput(this);
	}
	T evaluate() {
		return m_val;
	}
	void setInput(T val) {
		m_val = val;
	}
};
