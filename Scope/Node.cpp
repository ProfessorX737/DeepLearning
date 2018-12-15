#include "Node.h"
#include "Tensor.h"
#include "Graph.h"

Node::Node(Graph& graph, const std::string& class_name) {
	// set class_id
	auto it = classIdMap.find(class_name);
	if (it != classIdMap.end()) {
		class_id_ = it->second;
	}
	else {
		class_id_ = classIdMap.size();
		classIdMap[class_name] = class_id_;
	}
	class_name_ = class_name;
	// set scope_id
	id_ = graph.getUniqueId();
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
