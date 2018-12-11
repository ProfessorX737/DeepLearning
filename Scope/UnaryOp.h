#pragma once
#include "Op.h"

class UnaryOp : public Op<1> {
public:
	UnaryOp(const Node& operand, const std::string& class_name, Graph& graph) : Op({ {&operand} }, class_name, graph) {}
private:
	virtual void unaryOp(const Tensor& operand, Tensor& out) const = 0;
	void op(const std::array<Tensor,1>& in, Tensor& out) const override {
		unaryOp(in[0], out);
	}
};

//#pragma once
//#include "Node.h"
//#include "Tensor.h"
//
//
//class UnaryOp : public Node {
//public:
//	UnaryOp(Graph& graph, const Node& operand, const std::string& class_name) : Node(class_name, graph) {
//		children_.push_back(&operand);
//	}
//	void eval(Tensor& out) const override {
//		Tensor in;
//		children_[0]->eval(in);
//		unaryOp(in, out);
//	}
//	void eval(std::unordered_map<int, Tensor>& nodeTensorMap, Tensor& out) const override {
//		Tensor in;
//		auto it = nodeTensorMap.find(children_[0]->getId());
//		if (it == nodeTensorMap.end()) {
//			children_[0]->eval(nodeTensorMap,in);
//			nodeTensorMap[children_[0]->getId()] = in;
//		}
//		else {
//			in = it->second;
//		}
//		unaryOp(in,out);
//	}
//private:
//	virtual void unaryOp(const Tensor& operand, Tensor& out) const = 0;
//};
