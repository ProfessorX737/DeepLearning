#pragma once
#include "Op.h"

template<typename T>
class UnaryOp : public Op<T,1> {
public:
	UnaryOp(Graph& graph, NodePtr& operand, const std::string& class_name) : Op<T,1>(graph, { {operand} }, class_name) {
	}
private:
	virtual void unaryOp(const Tensor& operand, Tensor& out) const = 0;
	void op(const std::array<Tensor,1>& in, Tensor& out) const override {
		unaryOp(in[0], out);
	}
};
