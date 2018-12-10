#pragma once
#include "BinaryOp.h"

template<typename T>
class Add : public BinaryOp {
public:
	Add(Graph& graph, const Node& a, const Node& b) : BinaryOp(a, b, "Add", graph) {}
	~Add() {}
private:
	void binaryOp(const Tensor& a, const Tensor& b, Tensor& out) const override {
		CHECK_GE(a.numDims(), 1);
		CHECK_GE(b.numDims(), 1);
		CHECK(a.shape().isSameShape(b.shape())) << "Adding two different shapes: "
			<< a.dimString() << " vs " << b.dimString();
		TensorShape out_shape;
		out_shape = (a.numElements() > b.numElements()) ? a.shape() : b.shape();
		out.init(out_shape, a.dataType());

		auto vecA = a.asVec<T>();
		auto vecB = b.asVec<T>();
		auto vecOut = out.asVec<T>();
		vecOut = vecA + vecB;
	}
};
