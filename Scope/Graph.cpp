#include "Graph.h"
#include "Node.h"
#include "Tensor.h"
#include <iostream>

//size_t hashFn(Node* n) {
//	std::hash<int> intHasher;
//	return intHasher(n->getId());
//}
//
//bool keyCompFn(Node* n1, Node* n2) {
//	std::cout << "calling key compare func" << std::endl;
//	return n1 == n2;
//}

Graph::Graph() : numNodes_(0) {}
Graph::~Graph() {}

int Graph::getUniqueId() {
	return numNodes_++;
}

void Graph::eval(const Node& fetch, Tensor& out) {
	fetch.eval(out);
}

void Graph::eval(const std::unordered_map<int, Tensor>& feed_inputs, const std::vector<Node>& fetch_outputs, std::vector<Tensor>& out) {

	int maxDepth = 0;
	for (int i = 0; i < fetch_outputs.size(); i++) {
		int nodeId = fetch_outputs[i].getId();
		auto it = nodeDepthMap_.find(nodeId);
		if (it == nodeDepthMap_.end()) LOG(FATAL) << "fetch output node with id " << nodeId << ", is not in the graph";
		if (it->second > maxDepth) maxDepth = it->second;
	}
	std::vector<Node*>* levels = new std::vector<Node*>[maxDepth];

	for (int i = 0; i < fetch_outputs.size(); i++) {
		std::vector<Node*> conn;
		fetch_outputs[i].collect(conn);
		for (Node* n : conn) {
			levels[nodeDepthMap_[n->getId()]].push_back(n);
		}
	}

	std::unordered_map<int, Tensor> nodeTensorMap = feed_inputs;

	for (int i = 0; i < maxDepth; i++) {
		for (Node* n : levels[i]) {
			Tensor out;
			n->eval(nodeTensorMap, out);
			nodeTensorMap[n->getId()] = std::move(out);
		}
	}

	for (int i = 0; i < fetch_outputs.size(); i++) {
		out.push_back(nodeTensorMap[fetch_outputs[i].getId()]);
	}

	delete levels;
}

void Graph::addNode(const Node& node) {
	nodeDepthMap_[node.getId()] = node.depth();
}
