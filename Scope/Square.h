#pragma once

#include "UnaryOp.h"

template<typename T>
class Square : public UnaryOp {
public:
	Square(Graph& graph, const Node& in) : UnaryOp(graph, in, "Square") {}
	void unaryOp(const Tensor& in, Tensor& out) const override {
		TensorShape outShape = in.shape();
		out.init(outShape, in.dataType());
		auto outVec = out.asVec<T>();
		outVec = in.asVec<T>().array().square().matrix();
	}
};
