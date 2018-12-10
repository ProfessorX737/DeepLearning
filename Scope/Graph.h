#pragma once

#include <unordered_map>
#include <vector>
#include <functional>
#include "logging.h"

class Node;
class Tensor;

class Graph {
public:
	Graph();
	~Graph();
	void eval(const Node& fetch_output, Tensor& out);
	void eval(const std::unordered_map<int, Tensor>& feed_inputs, const std::vector<Node>& fetch_outputs, std::vector<Tensor>& out);
	void addNode(const Node& node);

private:
	int numNodes_;

	friend class Node;
	int getUniqueId();
	std::unordered_map<int, int> nodeDepthMap_;

};

