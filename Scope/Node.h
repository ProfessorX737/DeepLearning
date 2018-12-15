#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <array>
#include "types.h"
#include <memory>
#include <set>
#include <functional>

class Tensor;
class Graph;
class Node;

typedef std::shared_ptr<Node> NodePtr;
static std::unordered_map<std::string, uint32> classIdMap;

class Node {
	
public:
	Node(Graph& graph, const std::string& class_name);

	virtual void eval(Tensor& out) const = 0;
	virtual void eval(std::unordered_map<int,Tensor>& nodeTensorMap, Tensor& out) const { eval(out); }
	virtual DataType dataType() const = 0;
	//virtual bool deriv(Tensor& out) const = 0;

	int getClassId() const { return class_id_; }
	int getId() const { return id_; }
	int numChildren() const { return children_.size(); }

protected:
	friend class Graph;
	std::string class_name_;
	std::vector<NodePtr> children_;
private:
	// class id is unique for different class types
	int class_id_;
	// node id is unique for class instances within the given graph
	int id_;
};

