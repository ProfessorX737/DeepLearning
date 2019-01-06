#pragma once
#include "Op.h"

class UnaryOp : public Op<1> {
public:
	UnaryOp(Graph& graph, NodePtr& operand, const std::string& class_name) : Op(graph, { {operand} }, class_name) {
	}
private:
	virtual void unaryOp(const Tensor& operand, Tensor& out) const = 0;
	void op(const std::array<Tensor,1>& in, Tensor& out) const override {
		unaryOp(in[0], out);
	}
};
