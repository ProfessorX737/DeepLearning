#pragma once

#include <unordered_map>
#include <vector>
#include <functional>
#include "logging.h"
#include <set>

class Node;
class Tensor;

class Graph {
public:
	typedef std::set<Node*, std::function<bool(Node*, Node*)>> Set;

	Graph();
	~Graph();

	void eval(const Node& fetch_output, Tensor& out);
	void eval(const std::unordered_map<int, Tensor>& feed_inputs, const std::vector<Node*>& fetch_outputs, std::vector<Tensor>& out);

	// deprecated, use eval(). here for reference
	void eval_(const std::unordered_map<int, Tensor>& feed_inputs, const std::vector<Node*>& fetch_outputs, std::vector<Tensor>& out);

	int addNode(Node* node);

private:
	friend class Node;
	int numNodes_;
	std::unordered_map<int, int> nodeDepthMap_;
	std::unordered_map<int, Set> nodeTreeMap_;

	int getUniqueId();

};

