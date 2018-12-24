#pragma once
#include "BinaryOp.h"

template<typename T>
class AddOp : public BinaryOp {
public:
	AddOp(Graph& graph, NodePtr& a, NodePtr& b) : BinaryOp(graph, a, b, "Add") {}
	~AddOp() {}
private:
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
		vecOut = vecA + vecB;
	}
	void deriv(Tensor& dx, const std::array<Tensor, 2>& in, int wrtIdx) const {
		DCHECK((wrtIdx == 0) || (wrtIdx == 1));
		out = dx;
	}
};

inline NodePtr Add(Graph& graph, NodePtr a, NodePtr b) {
	NodePtr ret;
	NUMBER_TYPE_CASES(a->dataType(), ret = std::make_shared<AddOp<T>>(graph, a, b));
	return ret;
}
