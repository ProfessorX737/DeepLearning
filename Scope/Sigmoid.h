//
//  Sigmoid.h
//  DeepLearning
//
//  Created by Xavier Poon on 2/01/2019.
//  Copyright © 2019 CreativityInk. All rights reserved.
//

#pragma once
#include "UnaryOp.h"

template<typename T>
class SigmoidOp : public UnaryOp {
public:
    SigmoidOp(Graph& graph, NodePtr operand) : UnaryOp(graph,operand,"Sigmoid") {}
    void unaryOp(const Tensor& operand, Tensor& out) const override {
        out.init(operand.shape(),operand.dataType());
        out.asVec<T>().array() = 1/(1 + (operand.asVec<T>().array() * static_cast<T>(-1)).exp());
    }
	void deriv(Tensor& dx, const std::array<Tensor, 1>& in, int wrtIdx) const override {
		CHECK_EQ(wrtIdx, 0);
		Tensor sig(in[0].shape(), in[0].dataType());
        sig.asVec<T>().array() = 1/(1 + (in[0].asVec<T>().array() * static_cast<T>(-1)).exp());
        dx.asVec<T>().array() = dx.asVec<T>().array () * (sig.asVec<T>().array() * (1 - sig.asVec<T>().array()));
	}
};

inline NodePtr Sigmoid(Graph& graph, NodePtr operand) {
    NodePtr ret;
    NUMBER_TYPE_CASES(operand->dataType(),ret = std::make_shared<SigmoidOp<T>>(graph,operand));
    return ret;
}
