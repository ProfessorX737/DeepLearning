//
//  Multiply.h
//  DeepLearning
//
//  Created by Xavier Poon on 25/12/2018.
//  Copyright © 2018 CreativityInk. All rights reserved.
//
#pragma once

#include "UnaryOp.h"
#include "BinaryOp.h"

template<typename T>
class ScalarMultiplyOp : public UnaryOp {
public:
	ScalarMultiplyOp(Graph& graph, NodePtr& operand, T scalar)
    : UnaryOp(graph, operand, "MultiplyOp"), scalar_(scalar) {}
    
	void unaryOp(const Tensor& in, Tensor& out) const override {
		scalarMultiply(in, scalar_, out);
	}
	static void scalarMultiply(const Tensor& in, const T scalar, Tensor& out) {
        DCHECK_EQ(DataTypeToEnum<T>::v(),in.dataType());
        out.init(in.shape(),in.dataType());
        out.asVec<T>() = (in.asVec<T>().array() * scalar).matrix();
	}
    void deriv(Tensor& dx, const std::array<Tensor, 1>& in, int wrtIdx,
               const std::unordered_map<int,Tensor>& nodeTensorMap) const override {
		dx.asVec<T>() = (dx.asVec<T>().array() * scalar_).matrix();
	}
private:
    T scalar_;
};

template<typename T>
class CWiseMultiplyOp : public BinaryOp {
public:
	CWiseMultiplyOp(Graph& graph, NodePtr& a, NodePtr& b)
		: BinaryOp(graph, a, b, "CWiseMultiply") {}

	// allows broadcasting for scalars only
	void binaryOp(const Tensor& a, const Tensor& b, Tensor& out) const override {
		cWiseMultiply(a, b, out);
	}
	static void cWiseMultiply(const Tensor& a, const Tensor& b, Tensor& out) {
		DCHECK_EQ(DataTypeToEnum<T>::v(), a.dataType());
		if (a.numElements() == 1) {
			out.init(b.shape(), b.dataType());
			out.asVec<T>() = (b.asVec<T>().array() * a.data<T>()[0]).matrix();
		}
		else if (b.numElements() == 1) {
			out.init(a.shape(), a.dataType());
			out.asVec<T>() = (a.asVec<T>().array() * b.data<T>()[0]).matrix();
		}
		else {
            CHECK(a.hasSameShape(b)) << "operand tensors for cwise multiplication must have same shapes: " << a.dimString() << " vs " << b.dimString();
			out.init(a.shape(), a.dataType());
			out.asVec<T>() = (a.asVec<T>().array() * b.asVec<T>().array()).matrix();
		}
	}
	void deriv(Tensor& dx, const std::array<Tensor, 2>& in, int wrtIdx,
               const std::unordered_map<int,Tensor>& nodeTensorMap) const override {
		DCHECK(((wrtIdx == 0) || (wrtIdx == 1)));
        helperMultiply(dx, in[1 - wrtIdx]);
	}
private:
    // more optimized than cWiseMultiply by avoiding unnecessary heap allocations
    static void helperMultiply(Tensor& inout, const Tensor& other) {
        if(other.numElements() == 1) {
            inout.asVec<T>().array() = inout.asVec<T>().array() * other.data<T>()[0];
        } else if(inout.numElements() == 1) {
            Tensor scalar = inout;
            inout.init(other.shape(),other.dataType());
            inout.asVec<T>().array() = other.asVec<T>().array() * scalar.data<T>()[0];
        } else {
            CHECK(inout.hasSameShape(other));
            inout.asVec<T>().array() = inout.asVec<T>().array() * other.asVec<T>().array();
        }
    }
};

template<typename T>
inline NodePtr Multiply(Graph& graph, NodePtr operand, const T scalar) {
    NodePtr res;
	res = std::make_shared<ScalarMultiplyOp<T>>(graph, operand, scalar);
    return res;
}

inline NodePtr Multiply(Graph& graph, NodePtr a, NodePtr b) {
	NodePtr res;
	NUMBER_TYPE_CASES(a->dataType(), res = std::make_shared<CWiseMultiplyOp<T>>(graph, a, b));
	return res;
}