#pragma once
#include "UnaryOp.h"

template<typename T>
class TanhOp : public UnaryOp<T> {
public:
    TanhOp(Graph& graph, NodePtr& in) : UnaryOp<T>(graph, in, "Tanh") {}

	void unaryOp(const Tensor& in, Tensor& out) const override {
		out.init(in.shape(), in.dataType());
		auto outVec = out.asVec<T>();
		outVec = in.asVec<T>().array().tanh().matrix();
	}
    void deriv(Tensor& dx, DerivContext<1>& ctx) const override {
		CHECK_EQ(ctx.wrtIdx, 0);
        auto it = ctx.nodeTensorMap.find(UnaryOp<T>::getId());
        if(it == ctx.nodeTensorMap.end()) {
            LOG(FATAL) << "Could not find pre evaluated tanh for tanh derivative";
        }
        Tensor tanh = it->second;
        dx.asVec<T>().array() = dx.asVec<T>().array() * (1 - tanh.asVec<T>().array().square());
    }
//	void deriv(Tensor& dx, const std::array<Tensor, 1>& in, int wrtIdx,
//               const std::unordered_map<int,Tensor>& nodeTensorMap) const override {
//		CHECK_EQ(wrtIdx, 0);
//        auto it = nodeTensorMap.find(UnaryOp<T>::getId());
//        if(it == nodeTensorMap.end()) {
//            LOG(FATAL) << "Could not find pre evaluated tanh for tanh derivative";
//        }
//        Tensor tanh = it->second;
//        dx.asVec<T>().array() = dx.asVec<T>().array() * (1 - tanh.asVec<T>().array().square());
//	}
};

inline NodePtr Tanh(Graph& graph, NodePtr in) {
	NodePtr ret;
	NUMBER_TYPE_CASES(in->dataType(), ret = std::make_shared<TanhOp<T>>(graph, in));
	return ret;
}
