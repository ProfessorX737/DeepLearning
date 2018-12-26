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
    
    virtual void eval(std::unordered_map<int,Tensor>& nodeTensorMap, Tensor& out) const = 0;
    
    // override if node can be an operand to another node
    virtual DataType dataType() const;
    
    // only to be overridden in class Op
    virtual void evalDeriv(Tensor& dx, std::unordered_map<int,Tensor>& nodeTensorMap,
                           const std::vector<int>& path, const int pathIndex) const {}

    // not to be overridden in any new nodes created
    virtual void eval(Tensor& out) const;
    
    void eval(std::unordered_map<int,Tensor>& nodeTensorMap) const;
    
    // requires override only for Variable nodes
    void collectPaths(std::vector<std::vector<int>>& paths) const;
    
    // helper for collectPaths
	virtual void collectPaths(std::vector<int>& curr, std::vector<std::vector<int>>& paths) const;

	int getClassId() const { return class_id_; }
	int getId() const { return id_; }
	int numChildren() const { return static_cast<int>(children_.size()); }

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

