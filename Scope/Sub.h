#pragma once
#include "BinaryOp.h"

template<typename T>
class SubOp : public BinaryOp {
public:
	SubOp(Graph& graph, NodePtr& a, NodePtr& b) : BinaryOp(graph, a, b, "Sub") {}
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
	bool deriv(Tensor& out) const override {
		Tensor t1, t2;
		bool lhs = children_[0]->deriv(t1);
		bool rhs = children_[1]->deriv(t2);
		if (lhs && rhs) {
			binaryOp(t1, t2, out);
			return true;
		}
		else if (lhs) {
			out = t1;
			return true;
		}
		else if (rhs) {
			out = t2;
			out.asVec<T>() = (out.asVec<T>().array() * -1).matrix();
			return true;
		}
		else {
			binaryOp(t1, t2, out);
			return false;
		}
	}
};

inline NodePtr Sub(Graph& graph, NodePtr a, NodePtr b) {
	NodePtr ret;
	NUMBER_TYPE_CASES(a->dataType(), ret = std::make_shared<SubOp<T>>(graph, a, b));
	return ret;
}