#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <array>
#include "types.h"

class Tensor;

static std::unordered_map<std::string, uint32> classIdMap;

class Node {
public:
	Node(const std::string& class_name) {
		auto it = classIdMap.find(class_name);
		if (it != classIdMap.end()) {
			class_id_ = it->second;
		}
		else {
			class_id_ = classIdMap.size();
			classIdMap[class_name] = class_id_;
		}
	}
	virtual void eval(Tensor& out) = 0;
	virtual void feed(const Tensor& tensor) {}

	void collect(std::vector<Node*>& conn_nodes, uint32 class_id) {
		if (class_id_ == class_id) conn_nodes.push_back(this);
		for (Node* node : children_) node->collect(conn_nodes, class_id);
	}
protected:
	std::vector<Node*> children_;
private:
	uint32 class_id_;
};