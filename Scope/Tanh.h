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
		CHECK_EQ(wrtIdx, 0);
		//dx.multiply<T>(1 - in[0].asVec<T>().array().tanh().square());
		Tensor dtanh(in[0].shape(), in[0].dataType());
		dtanh.asVec<T>() = (1 - in[0].asVec<T>().array().tanh().square()).matrix();
		dx = dx.cWiseMult<T>(dtanh);
	}
};

inline NodePtr Tanh(Graph& graph, NodePtr in) {
	NodePtr ret;
	NUMBER_TYPE_CASES(in->dataType(), ret = std::make_shared<TanhOp<T>>(graph, in));
	return ret;
}
