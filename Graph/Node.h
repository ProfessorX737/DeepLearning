#pragma once
#include "Graph.h"

template<typename T>
class Node {
private:
	unsigned int m_id;
public:
	typedef T value_type;
	Node(Graph<Node<T>>& g) {
		g.addNode(this);
	}
	int getId() {
		return m_id;
	}
	void setId(unsigned int id) {
		m_id = id;
	}
	virtual T evaluate() = 0;
};
