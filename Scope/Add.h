#pragma once
#include "BinaryOp.h"
#include "Broadcast.h"

template<typename T>
class AddOp : public BinaryOp<T> {
public:
	AddOp(Graph& graph, NodePtr& a, NodePtr& b) : BinaryOp<T>(graph, a, b, "Add") {}
	~AddOp() {}
private:
	void binaryOp(const Tensor& a, const Tensor& b, Tensor& out) const override {
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
                    out.tensorPadLeft<T,Tensor::MAX_DIMS>() = A + B.broadcast(multDims);
                }
            } else {
                out.init(b.shape(), b.dataType());
                out.tensorPadLeft<T,Tensor::MAX_DIMS>() = A.broadcast(multDims) + B;
            }
        } else {
    		out.init(a.shape(), a.dataType());
            out.tensorPadLeft<T,Tensor::MAX_DIMS>() = A + B;
        }
	}
    void deriv(Tensor& dx, DerivContext<2>& ctx) const override {
		DCHECK(((ctx.wrtIdx == 0) || (ctx.wrtIdx == 1)));
        
    }

};

inline NodePtr Add(Graph& graph, NodePtr a, NodePtr b) {
	NodePtr ret;
	NUMBER_TYPE_CASES(a->dataType(), ret = std::make_shared<AddOp<T>>(graph, a, b));
	return ret;
}
