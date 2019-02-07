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
    
    // evaluates this node by recursively evaluating all the children of this node
    // returns the output Tensor of this node @out
    // returns the all the tensors that are required to evaluate this node @nodeTensorMap
    virtual void eval(std::unordered_map<int,Tensor>& nodeTensorMap, Tensor& out) = 0;
    
    // override if node can be an operand to another node
    virtual DataType dataType() const;
    
    // calculates the derivatives of this node with respect to each variable that make up this node.
    // takes in the nodeTensorMap that is created from the eval(&nodeTensorMap,&out) function.
    // returns a vector of gradients @inOutGrads.
    void evalGradients(std::unordered_map<int,Tensor>& nodeTensorMap, const std::vector<std::vector<int>>& paths,
                       std::vector<Tensor>& inOutGrads, int batchIndex);
    
    // only to be overridden in class Op
    // helper recursive function for evalGradients
    virtual void evalDeriv(Tensor& dx, std::unordered_map<int,Tensor>& nodeTensorMap,
                           const std::vector<int>& path, const int pathIndex, int batchIndex) {}

    // returns the output result of this node @out
    // not to be overridden in any new nodes created
    virtual void eval(Tensor& out);
    
    // returns the all the tensors and their correspondings nodes that are required
    // to evaluate this node @nodeTensorMap
    void eval(std::unordered_map<int,Tensor>& nodeTensorMap);
    
    // returns a vector of paths-to-variables
    void collectPaths(std::vector<std::vector<int>>& outPaths, std::vector<Tensor>& outVariables) const;
    
    // helper for collectPaths
    // requires override only for Variable nodes
    virtual void collectPaths(std::vector<int>& curr, std::vector<std::vector<int>>& paths,
                              std::vector<Tensor>& outVariables) const;

	int getClassId() const { return class_id_; }
    std::string getClassName() const { return class_name_; }
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

