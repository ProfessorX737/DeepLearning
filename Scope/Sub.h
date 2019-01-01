#pragma once
#include "BinaryOp.h"

template<typename T>
class SubOp : public BinaryOp {
public:
	SubOp(Graph& graph, NodePtr& a, NodePtr& b) : BinaryOp(graph, a, b, "Sub") {}
	void binaryOp(const Tensor& a, const Tensor& b, Tensor& out) const override {
		subtract(a, b, out);
	}
	static void subtract(const Tensor& a, const Tensor& b, Tensor& out) {
		CHECK_GE(a.numDims(), 1);
		CHECK_GE(b.numDims(), 1);
		CHECK(a.shape().isSameShape(b.shape())) << "Adding two different shapes: "
			<< a.dimString() << " vs " << b.dimString();
		TensorShape outShape = a.shape();
		out.init(outShape, a.dataType());

		auto vecA = a.asVec<T>();
		auto vecB = b.asVec<T>();
		auto vecOut = out.asVec<T>();
		vecOut = vecA - vecB;
	}
	void deriv(Tensor& dx, const std::array<Tensor, 2>& in, int wrtIdx) const override {
		DCHECK(((wrtIdx == 0) || (wrtIdx == 1)));
		if (wrtIdx == 1) {
			dx = dx.scalarMult<T>(-1);
		}
	}
	
};

inline NodePtr Sub(Graph& graph, NodePtr a, NodePtr b) {
	NodePtr ret;
	NUMBER_TYPE_CASES(a->dataType(), ret = std::make_shared<SubOp<T>>(graph, a, b));
	return ret;
}