#pragma once
#include "Op.h"
#include <string>

class BinaryOp : public Op<2> {
public:
	BinaryOp(Graph& graph, NodePtr& left, NodePtr& right, const std::string& class_name) : Op(graph, { {left, right} }, class_name) {
	}
private:
	virtual void binaryOp(const Tensor& left, const Tensor& right, Tensor& out) const = 0;
	void op(const std::array<Tensor, 2>& in, Tensor& out) const override {
		CHECK_EQ(in[0].dataType(), in[1].dataType()) 
			<< "binary operands need to use the same data type: "
			<< in[0].dataType() << " vs " << in[1].dataType();
		binaryOp(in[0], in[1], out);
	}
};
