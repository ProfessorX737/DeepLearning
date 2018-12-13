#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <array>
#include "types.h"
#include "Graph.h"
#include <memory>
#include <set>

class Tensor;

static std::unordered_map<std::string, uint32> classIdMap;

class Node {
	
public:
	typedef std::set<Node*, std::function<bool(Node*, Node*)>> Set;
	Node(Graph& graph, const std::string& class_name);

	virtual void eval(Tensor& out) const = 0;
	void evaluate(const std::unordered_map<int, Tensor>& feed, Tensor& out) const;
	virtual void eval(std::unordered_map<int, Tensor>& nodeTensorMap, Tensor& out) const { eval(out); }

	int collect(Set& conn_nodes, std::unordered_map<int,int>& depthMap, int level = 0) const;

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
	int numChildren() const { return children_.size(); }

protected:
	friend class Graph;
	std::string class_name_;
	std::vector<const Node*> children_;
private:
	// class id is unique for different class types
	int class_id_;
	// node id is unique for class instances within the given graph
	int id_;
};

