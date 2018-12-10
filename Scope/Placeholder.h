#pragma once
#include "Node.h"
#include "logging.h"

class Placeholder : public Node {
public:
	Placeholder(Graph& graph) : Node("Placeholder",graph) {}
	void eval(Tensor& out) const override {
		LOG(FATAL) << "Placeholder with node id: " << getId() << ", needs to be fed a value";
	}
	void eval(std::unordered_map<int, Tensor>& nodeTensorMap, Tensor& out) const override {
		auto it = nodeTensorMap.find(children_[0]->getId());
		if (it == nodeTensorMap.end()) LOG(FATAL) << "Missing placeholder input";
		out = it->second;
	}

};
