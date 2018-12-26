#pragma once

#include <unordered_map>
#include <vector>
#include "logging.h"
#include <set>
#include <memory>

class Tensor;
class Node;

class Graph {
public:
	typedef std::shared_ptr<Node> NodePtr;
	//typedef std::set<Node*, std::function<bool(Node*, Node*)>> Set;
	Graph();
	~Graph();

	static void eval(NodePtr& fetch_output, Tensor& out);
    
	static void eval(const std::unordered_map<NodePtr,Tensor>& feed_inputs,
                     const std::vector<NodePtr>& fetch_outputs,
                     std::vector<Tensor>& out);
    
    static void eval(std::unordered_map<int,Tensor>& nodeTensorMap,
                     const std::unordered_map<NodePtr,Tensor>& feed_inputs,
                     const std::vector<NodePtr>& fetch_outputs,
                     std::vector<Tensor>& out);

	// deprecated, use eval(). here for reference
	//void eval_(const std::unordered_map<int, Tensor>& feed_inputs, const std::vector<Node*>& fetch_outputs, std::vector<Tensor>& out);

	//int addNode(Node* node);

private:
	friend class Node;
	int numNodes_;
	//std::unordered_map<int, int> nodeDepthMap_;
	//std::unordered_map<int, Set> nodeTreeMap_;

	int getUniqueId();

};

