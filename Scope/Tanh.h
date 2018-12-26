#pragma once
#include "UnaryOp.h"

template<typename T>
class TanhOp : public UnaryOp {
public:
    TanhOp(Graph& graph, NodePtr& in) : UnaryOp(graph, in, "Tanh") {}

	void unaryOp(const Tensor& in, Tensor& out) const override {
		out.init(in.shape(), in.dataType());
		auto outVec = out.asVec<T>();
		outVec = in.asVec<T>().array().tanh().matrix();
	}
	void deriv(Tensor& dx, const std::array<Tensor, 1>& in, int wrtIdx) const override {
        
	}
};

inline NodePtr Tanh(Graph& graph, NodePtr in) {
	NodePtr ret;
	NUMBER_TYPE_CASES(in->dataType(), ret = std::make_shared<TanhOp<T>>(graph, in));
	return ret;
}
