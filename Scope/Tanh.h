#pragma once
#include "UnaryOp.h"
#include <Eigen/Dense>

class Tensor;

template<typename T>
class Tanh : public UnaryOp {
public:
	Tanh(Graph& graph, const Node& in) : UnaryOp(in, "Tanh", graph) {}
	void unaryOp(const Tensor& in, Tensor& out) const override {
		TensorShape outShape = in.shape();
		out.init(outShape, in.dataType());
		auto outVec = out.asVec<T>();
		outVec = in.asVec<T>().array().tanh().matrix();
	}
};
