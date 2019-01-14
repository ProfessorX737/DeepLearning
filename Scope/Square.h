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
    
	void deriv(Tensor& dx, const std::array<Tensor, 1>& in, int wrtIdx,
               const std::unordered_map<int,Tensor>& nodeTensorMap) const override {
		DCHECK_EQ(wrtIdx, 0);
        CHECK(dx.hasSameShape(in[0])) << dx.dimString() << " vs " << in[0].dimString();
        dx.asVec<T>().array() = dx.asVec<T>().array() * (static_cast<T>(2) * in[0].asVec<T>().array());
	}
};

inline NodePtr Square(Graph& graph, NodePtr in) {
	NodePtr ret;
	NUMBER_TYPE_CASES(in->dataType(), ret = std::make_shared<SquareOp<T>>(graph, in));
	return ret;
}
