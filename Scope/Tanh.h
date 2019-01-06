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
	void deriv(Tensor& dx, const std::array<Tensor, 1>& in, int wrtIdx,
               const std::unordered_map<int,Tensor>& nodeTensorMap) const override {
		CHECK_EQ(wrtIdx, 0);
        auto it = nodeTensorMap.find(getId());
        if(it == nodeTensorMap.end()) {
            LOG(FATAL) << "Could not find pre evaluated tanh for tanh derivative";
        }
        Tensor tanh = it->second;
        dx.asVec<T>().array() = dx.asVec<T>().array() * (1 - tanh.asVec<T>().array().square());
	}
};

inline NodePtr Tanh(Graph& graph, NodePtr in) {
	NodePtr ret;
	NUMBER_TYPE_CASES(in->dataType(), ret = std::make_shared<TanhOp<T>>(graph, in));
	return ret;
}
