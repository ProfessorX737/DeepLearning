#include "Node.h"
#include "Tensor.h"
#include "Graph.h"
#include <iostream>

Node::Node(Graph& graph, const std::string& class_name) {
	// set class_id
	auto it = classIdMap.find(class_name);
	if (it != classIdMap.end()) {
		class_id_ = it->second;
	}
	else {
		class_id_ = static_cast<int>(classIdMap.size());
		classIdMap[class_name] = class_id_;
	}
	class_name_ = class_name;
	// set scope_id
	id_ = graph.getUniqueId();
}

void Node::eval(Tensor& out) const {
    std::unordered_map<int,Tensor> nodeTensorMap;
    eval(nodeTensorMap,out);
};

void Node::eval(std::unordered_map<int,Tensor>& nodeTensorMap) const {
    Tensor out;
    eval(nodeTensorMap,out);
}

DataType Node::dataType() const {
    LOG(ERROR) << "dataType() function not implemented in node " << class_name_;
    return DT_INVALID;
}

void Node::collectPaths(std::vector<std::vector<int>>& paths) const {
    std::vector<int> curr;
    collectPaths(curr, paths);
}

void Node::collectPaths(std::vector<int>& curr, std::vector<std::vector<int>>& paths) const {
	for (int i = 0; i < children_.size(); i++) {
		std::vector<int> pathToNext = curr;
		pathToNext.push_back(i);
		children_[i]->collectPaths(pathToNext,paths);
	}
}

//void Node::evaluate(const std::unordered_map<int, Tensor>& feed, Tensor& out) const {
//	std::unordered_map<int, Tensor> nodeTensorMap = std::move(feed);
//	eval(nodeTensorMap, out);
//}

//int Node::collect(Set& conn_nodes, std::unordered_map<int,int>& depthMap, int level) const {
//	int maxDepth = level;
//	conn_nodes.insert(this);
//	for (NodePtr node : children_) {
//		int depth = node->collect(conn_nodes, depthMap, level + 1);
//		if (depth > maxDepth) maxDepth = depth;
//	}
//	depthMap[id_] = maxDepth - level;
//	return maxDepth;
//}
