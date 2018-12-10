#include "Node.h"

Node::Node(const std::string& class_name, Graph& graph) {
	// set class_id
	auto it = classIdMap.find(class_name);
	if (it != classIdMap.end()) {
		class_id_ = it->second;
	}
	else {
		class_id_ = classIdMap.size();
		classIdMap[class_name] = class_id_;
	}
	// set scope_id
	id_ = graph.getUniqueId();
	graph.addNode(*this);
}

void Node::collect(std::vector<Node*>& conn_nodes) const {
	conn_nodes.push_back(const_cast<Node*>(this));
	for (const Node* node : children_) node->collect(conn_nodes);
}

void Node::collect(std::vector<Node*>& conn_nodes, int class_id) const {
	if (class_id_ == class_id) conn_nodes.push_back(const_cast<Node*>(this));
	for (const Node* node : children_) node->collect(conn_nodes, class_id);
}
