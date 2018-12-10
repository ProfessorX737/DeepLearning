#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <array>
#include "types.h"
#include "Graph.h"
#include <memory>

class Tensor;

static std::unordered_map<std::string, uint32> classIdMap;

class Node {
	
public:
	Node(const std::string& class_name, Graph& graph);

	virtual void eval(Tensor& out) const = 0;
	virtual void eval(std::unordered_map<int, Tensor>& nodeTensorMap, Tensor& out) const { eval(out); }

	void collect(std::vector<Node*>& conn_nodes) const;
	void collect(std::vector<Node*>& conn_nodes, int class_id) const;
	void collect(std::unordered_map<int, int>& nodeLevelMap, int level) const {
		auto it = nodeLevelMap.find(id_);
		if (it == nodeLevelMap.end()) {
			nodeLevelMap.insert({ id_,level });
		}
		else {
			if (level > it->second) nodeLevelMap[id_] = level;
		}
		for (const Node* n : children_) {
			n->collect(nodeLevelMap, level+1);
		}
	}

	int depth(int level = 0) const {
		int maxDepth = level;
		for (const Node* n : children_) {
			int depth = n->depth(level+1);
			if (depth > maxDepth) maxDepth = depth;
		}
		return maxDepth;
	}

	int getClassId() const { return class_id_; }
	int getId() const { return id_; }

protected:
	friend class Graph;
	std::vector<const Node*> children_;
private:
	// class id is unique for different class types
	int class_id_;
	// node id is unique for class instances within the given graph
	int id_;
};

