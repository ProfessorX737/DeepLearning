#pragma once

#include "UnaryOp.h"

template<typename T>
class SquareOp : public UnaryOp<T> {
public:
	SquareOp(Graph& graph, NodePtr& in) : UnaryOp<T>(graph, in, "Square") {}
    
	void unaryOp(const Tensor& in, Tensor& out) const override {
		TensorShape outShape = in.shape();
		out.init(outShape, in.dataType());
		auto outVec = out.asVec<T>();
		outVec = in.asVec<T>().array().square().matrix();
	}
    void deriv(Tensor& dx, DerivContext<1>& ctx) const override {
		DCHECK_EQ(ctx.wrtIdx, 0);
        CHECK(dx.hasSameShape(ctx.operands[0])) << dx.dimString() << " vs " << ctx.operands[0].dimString();
        dx.asVec<T>().array() = dx.asVec<T>().array() * (static_cast<T>(2) * ctx.operands[0].asVec<T>().array());
    }
};

inline NodePtr Square(Graph& graph, NodePtr in) {
	NodePtr ret;
	NUMBER_TYPE_CASES(in->dataType(), ret = std::make_shared<SquareOp<T>>(graph, in));
	return ret;
}
