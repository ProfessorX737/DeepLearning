#include "Graph.h"
#include "Tensor.h"
#include "Node.h"

size_t hashFn(NodePtr n) {
	std::hash<int> intHasher;
	return intHasher(n->getId());
}

bool keyCompFn(NodePtr n1, NodePtr n2) {
	return n1->getId() < n2->getId();
}

Graph::Graph() : numNodes_(0) {}
Graph::~Graph() {}

int Graph::getUniqueId() {
	return numNodes_++;
}

void Graph::eval(NodePtr fetch, Tensor& out) {
	fetch->eval(out);
}

void Graph::eval(const std::unordered_map<NodePtr,Tensor>& feed_inputs,
                 const std::vector<NodePtr>& fetch_outputs) {
    std::vector<Tensor> out;
    eval(feed_inputs,fetch_outputs,out);
}

void Graph::eval(const std::unordered_map<NodePtr,Tensor>& feed_inputs,
                 const std::vector<NodePtr>& fetch_outputs,
                 std::vector<Tensor>& out) {
    std::unordered_map<int, Tensor> nodeTensorMap;
    eval(nodeTensorMap,feed_inputs,fetch_outputs,out);
}

void Graph::eval(std::unordered_map<int,Tensor>& nodeTensorMap,
                 const std::unordered_map<NodePtr,Tensor>& feed_inputs,
                 const std::vector<NodePtr>& fetch_outputs,
                 std::vector<Tensor>& out) {
	int numOut = static_cast<int>(fetch_outputs.size());
	for (auto it : feed_inputs) {
		nodeTensorMap[it.first->getId()] = std::move(it.second);
	}
	for (int i = 0; i < numOut; i++) {
		Tensor res;
		fetch_outputs[i]->eval(nodeTensorMap, res);
		out.push_back(std::move(res));
	}
}

