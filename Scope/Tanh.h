#pragma once
#include "UnaryOp.h"

template<typename T>
class TanhOp : public UnaryOp {
public:
	TanhOp(Graph& graph, NodePtr& in) : UnaryOp(graph, in, "Tanh") {}
	void unaryOp(const Tensor& in, Tensor& out) const override {
		TensorShape outShape = in.shape();
		out.init(outShape, in.dataType());
		auto outVec = out.asVec<T>();
		outVec = in.asVec<T>().array().tanh().matrix();
	}
};

inline NodePtr Tanh(Graph& graph, NodePtr in) {
	NodePtr ret;
	NUMBER_TYPE_CASES(in->dataType(), ret = std::make_shared<TanhOp<T>>(graph, in));
	return ret;
}
