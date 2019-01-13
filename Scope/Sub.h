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
        auto A = a.tensorPadLeft<T,Tensor::MAX_DIMS>();
        auto B = b.tensorPadLeft<T,Tensor::MAX_DIMS>();
        if(!a.hasSameShape(b)) {
            Eigen::array<int,Tensor::MAX_DIMS> multDims;
            if(!BCast::multDimsPadLeft(multDims, a.shape(), b.shape())) {
                if(!BCast::multDimsPadLeft(multDims, b.shape(), a.shape())) {
                   LOG(FATAL) << "operands to Add operation are not broadcast compatible";
                } else {
                    out.init(a.shape(), a.dataType());
                    out.tensorPadLeft<T,Tensor::MAX_DIMS>() = A - B.broadcast(multDims);
                }
            } else {
                out.init(b.shape(), b.dataType());
                out.tensorPadLeft<T,Tensor::MAX_DIMS>() = A.broadcast(multDims) - B;
            }
        } else {
    		out.init(a.shape(), a.dataType());
            out.tensorPadLeft<T,Tensor::MAX_DIMS>() = A - B;
        }
	}
	void deriv(Tensor& dx, const std::array<Tensor, 2>& in, int wrtIdx,
               const std::unordered_map<int,Tensor>& nodeTensorMap) const override {
		DCHECK(((wrtIdx == 0) || (wrtIdx == 1)));
		if (wrtIdx == 1) {
            dx.asVec<T>().array() = dx.asVec<T>().array() * static_cast<T>(-1);
		}
	}
	
};

inline NodePtr Sub(Graph& graph, NodePtr a, NodePtr b) {
	NodePtr ret;
	NUMBER_TYPE_CASES(a->dataType(), ret = std::make_shared<SubOp<T>>(graph, a, b));
	return ret;
}