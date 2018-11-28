#pragma once
#include <vector>
#include <unordered_map>
#include "Node.h"

template<class Node> 
class Graph {
private:
	typedef std::unordered_map<unsigned int, Node*> umap;
	unsigned int m_numNodes;
	umap m_nodeMap;
	umap m_inputMap;
public:
//	typedef Node::value_type value_type;
	Graph() : m_numNodes(0) {

	}
	void addNode(Node* node) {
		node->setId(m_numNodes);
		m_nodeMap[m_numNodes] = node;
		m_numNodes++;
	}
	void addInput(Node* inputNode) {
		m_inputMap[inputNode->getId()] = inputNode;
	}
	void getInputs(Node& node, std::vector<Node*>& inputs) {
		node.collectInputs(inputs);
	}
	//type evaluate(Node* node) {
	//	m_nodeMap[node->getId()].evaluate();
	//}
};
