#pragma once

#include "UnaryOp.h"

template<typename T>
class SquareOp : public UnaryOp {
public:
	SquareOp(Graph& graph, NodePtr& in) : UnaryOp(graph, in, "Square") {}
	void unaryOp(const Tensor& in, Tensor& out) const override {
		TensorShape outShape = in.shape();
		out.init(outShape, in.dataType());
		auto outVec = out.asVec<T>();
		outVec = in.asVec<T>().array().square().matrix();
	}
};

inline NodePtr Square(Graph& graph, NodePtr in) {
	NodePtr ret;
	NUMBER_TYPE_CASES(in->dataType(), ret = std::make_shared<SquareOp<T>>(graph, in));
	return ret;
}
