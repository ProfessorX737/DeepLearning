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

void Graph::eval(const std::unordered_map<NodePtr,Tensor>& feed_inputs, const std::vector<NodePtr>& fetch_outputs,
                 std::vector<Tensor>& out) {
//	int numOut = static_cast<int>(fetch_outputs.size());
//	std::unordered_map<int, Tensor> nodeTensorMap;
//	for (auto it : feed_inputs) {
//		nodeTensorMap[it.first->getId()] = std::move(it.second);
//	}
//	for (int i = 0; i < numOut; i++) {
//		Tensor res;
//		fetch_outputs[i]->eval(nodeTensorMap, res);
//		out.push_back(std::move(res));
//	}
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

//void Graph::eval_(const std::unordered_map<int, Tensor>& feed_inputs, const std::vector<Node::NodePtr>& fetch_outputs, std::vector<Tensor>& out) {
//
//	int numOut = fetch_outputs.size();
//	int* ids = new int[numOut];
//	for (int i = 0; i < numOut; i++) {
//		ids[i] = fetch_outputs[i]->getId();
//	}
//
//	int maxDepth = 0;
//	for (int i = 0; i < numOut; i++) {
//		auto it = nodeDepthMap_.find(ids[i]);
//		int depth;
//		if (it == nodeDepthMap_.end()) {
//			depth = addNode(fetch_outputs[i]);
//		} 
//		else {
//			depth = it->second;
//		}
//		if (depth > maxDepth) maxDepth = depth;
//	}
//
//	Set* levels = new Set[maxDepth+1];
//	for (int i = 0; i <= maxDepth; i++) {
//		levels[i] = Set(keyCompFn);
//	}
//
//	for (int i = 0; i < numOut; i++) {
//		for (Node::NodePtr n : nodeTreeMap_[ids[i]]) {
//			levels[nodeDepthMap_[n->getId()]].insert(n);
//		}
//	}
//
//	std::unordered_map<int, Tensor> nodeTensorMap = std::move(feed_inputs);
//
//	for (int i = 0; i <= maxDepth; i++) {
//		for (Node::NodePtr n : levels[i]) {
//			Tensor res;
//			n->eval(nodeTensorMap, res);
//			nodeTensorMap[n->getId()] = std::move(res);
//		}
//	}
//
//	for (int i = 0; i < fetch_outputs.size(); i++) {
//		out.push_back(nodeTensorMap[ids[i]]);
//	}
//
//	delete[] levels;
//	delete[] ids;
//}

//int Graph::addNode(Node::NodePtr node) {
//	Set conn(keyCompFn);
//	int depth = node->collect(conn, nodeDepthMap_);
//	nodeTreeMap_[node->getId()] = std::move(conn);
//	return depth;
//}
