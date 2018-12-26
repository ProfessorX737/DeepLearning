//
//  Multiply.h
//  DeepLearning
//
//  Created by Xavier Poon on 25/12/2018.
//  Copyright © 2018 CreativityInk. All rights reserved.
//
#pragma once

#include "UnaryOp.h"

template<typename T>
class MultiplyOp : public UnaryOp {
public:
	MultiplyOp(Graph& graph, NodePtr& operand, const T scalar)
    : UnaryOp(graph, operand, "MultiplyOp"), scalar_(scalar) {}
    
	void unaryOp(const Tensor& in, Tensor& out) const override {
        DCHECK_EQ(DataTypeToEnum<T>::v(),in.dataType());
        out.init(in.shape(),in.dataType());
        out.asVec<T>() = (in.asVec<T>().array() * scalar_).matrix();
	}
	void deriv(Tensor& dx, const std::array<Tensor, 1>& in, int wrtIdx) const override {
        dx.asVec<T>() = (dx.asVec<T>().array() * scalar_).matrix();
	}
private:
    T scalar_;
};

template<typename T>
inline NodePtr Multiply(Graph& graph, NodePtr operand, const T scalar) {
    NodePtr res;
    res = std::make_shared<MultiplyOp<T>>(graph,operand,scalar);
    return res;
}