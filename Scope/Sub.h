#pragma once

#include "BinaryOp.h"

template<typename T>
class Sub : public BinaryOp {
public:
	Sub(Graph& graph, const Node& a, const Node& b) : BinaryOp(graph, a, b, "Sub") {}
	void binaryOp(const Tensor& a, const Tensor& b, Tensor& out) const override {
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
};