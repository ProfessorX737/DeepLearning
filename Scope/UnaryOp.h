#pragma once
#include "Node.h"
#include "Tensor.h"


class UnaryOp : public Node {
public:
	UnaryOp(Graph& graph, const Node& operand, const std::string& class_name) : Node(class_name, graph) {
		children_.push_back(&operand);
	}
	void eval(Tensor& out) const override {
		Tensor in;
		children_[0]->eval(in);
		unaryOp(in, out);
	}
	void eval(std::unordered_map<int, Tensor>& nodeTensorMap, Tensor& out) const override {
		Tensor in;
		auto it = nodeTensorMap.find(children_[0]->getId());
		if (it == nodeTensorMap.end()) LOG(FATAL) << "Could not find operand tensor in nodeTensorMap";
		in = it->second;
		unaryOp(in,out);
	}
private:
	virtual void unaryOp(const Tensor& operand, Tensor& out) const = 0;
};
