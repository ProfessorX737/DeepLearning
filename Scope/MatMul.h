#pragma once
#include "BinaryOp.h"

template<typename T>
class MatMulOp : public BinaryOp<T> {
public:
	MatMulOp(Graph& graph, NodePtr& a, NodePtr& b) 
		: BinaryOp<T>(graph, a, b, "MatMul"), transA_(0), transB_(0) {}

	MatMulOp(Graph& graph, NodePtr& a, NodePtr& b, bool transA, bool transB)
		: BinaryOp<T>(graph, a, b, "MatMul"), transA_(transA), transB_(transB) {}

	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using ConstMatrixMap = Eigen::Map<const Matrix>;
	using MatrixMap = Eigen::Map<Matrix>;

	static ConstMatrixMap constEigenMatrixSlice(const Tensor& t, int slice) {
		return ConstMatrixMap(t.data<T>() + slice * t.dimSize(1) * t.dimSize(2),
			t.dimSize(1), t.dimSize(2));
	}

	static MatrixMap eigenMatrixSlice(Tensor& t, int slice) {
		return MatrixMap(t.data<T>() + slice * t.dimSize(1) * t.dimSize(2),
			t.dimSize(1), t.dimSize(2));
	}

	void binaryOp(const Tensor& a, const Tensor& b, Tensor& out) const override {
		matmult(a, b, out, transA_, transB_);
	}

	static void matmult(const Tensor& a, const Tensor& b, Tensor& out, bool transA = false, bool transB = false) {
		CHECK_EQ(a.numDims(), b.numDims()) 
			<< "a and b operands have different ndims: " 
			<< a.dimString() << " vs " << b.dimString();

		const int ndims = a.numDims();
		CHECK_GE(ndims,2) 
			<< "a and b operands need to have >= 2 dims";

		TensorShape out_shape;
		for (int i = 0; i < ndims - 2; i++) {
			CHECK_EQ(a.dimSize(i), b.dimSize(i))
				<< "dimSize(" << i << ") of" << a.dimString() << " and " 
				<< b.dimString() << " must be the same";
			out_shape.addDim(a.dimSize(i));
		}

		auto arows = a.dimSize(ndims - 2);
		auto acols = a.dimSize(ndims - 1);
		auto brows = b.dimSize(ndims - 2);
		auto bcols = b.dimSize(ndims - 1);

		// size of the batch dim is equal to product of all dim sizes except last two
        TensorShape::Dim nbatch = out_shape.numElements() == 0 ? 1 : out_shape.numElements();

		// make reshaped a & b that have most 3 dims
		Tensor rA, rB, rOut;
		rA.sharedCopyInit(a, TensorShape({ nbatch,arows,acols }));
		rB.sharedCopyInit(b, TensorShape({ nbatch,brows,bcols }));

		if (transA) std::swap(arows, acols);
		if (transB) std::swap(bcols, brows);

		CHECK_EQ(acols, brows) 
			<< "Inner dims must equal: " << a.dimString() << " vs " 
			<< b.dimString();

		out_shape.addDim(arows);
		out_shape.addDim(bcols);

		out.init(out_shape, a.dataType());

		rOut.sharedCopyInit(out, TensorShape({ nbatch,arows,bcols }));
		
		for (int i = 0; i < nbatch; i++) {
			auto matA = constEigenMatrixSlice(rA, i);
			auto matB = constEigenMatrixSlice(rB, i);
			auto matOut = eigenMatrixSlice(rOut, i);

			if (!transA) {
				if (!transB) {
					matOut.noalias() = matA * matB;
				}
				else {
					matOut.noalias() = matA * matB.transpose();
				}
			}
			else {
				if (!transB) {
					matOut.noalias() = matA.transpose() * matB;
				}
				else {
					matOut.noalias() = matA.transpose() * matB.transpose();
				}
			}
		}
	}

	// allows broadcasting for scalars
	static void mult(const Tensor& a, const Tensor& b, Tensor& out, bool transA = false, bool transB = false) {
		if (a.numElements() == 1) {
			out.init(b.shape(), b.dataType());
			out.asVec<T>() = (b.asVec<T>().array() * a.data<T>()[0]).matrix();
		}
		else if (b.numElements() == 1) {
			out.init(a.shape(), a.dataType());
			out.asVec<T>() = (a.asVec<T>().array() * b.data<T>()[0]).matrix();
		}
		else {
			matmult(a, b, out, transA, transB);
		}
	}
    void deriv(Tensor& dx, DerivContext<2>& ctx) const override {
		DCHECK(((ctx.wrtIdx == 0) || (ctx.wrtIdx == 1)));
		if (ctx.wrtIdx == 0) {
			matmult(dx, ctx.operands[1 - ctx.wrtIdx], dx, false, true);
		}
		else {
			matmult(ctx.operands[1 - ctx.wrtIdx], dx, dx, true, false);
		}
    }

	bool transA_;
	bool transB_;
};

inline NodePtr MatMul(Graph& graph, NodePtr a, NodePtr b) {
	NodePtr ret;
	NUMBER_TYPE_CASES(a->dataType(), ret = std::make_shared<MatMulOp<T> >(graph, a, b));
	return ret;
}

inline NodePtr MatMul(Graph& graph, NodePtr a, NodePtr b, bool transA, bool transB) {
	NodePtr ret;
	NUMBER_TYPE_CASES(a->dataType(), ret = std::make_shared<MatMulOp<T> >(graph, a, b,
		transA, transB));
	return ret;
}

//template<typename T>
//Tensor& operator*(Tensor& t, T scalar) {
//	t.asVec<T>() = t.asVec<T>().array() * scalar;
//	return t;
//}

//Tensor& operator*(const Tensor& t1, const Tensor& t2) {
//	if (t1.numElements() == 1) {
//		NUMBER_TYPE_CASES(t2.asVec<T>() = )
//	}
//	DCHECK_EQ(t1.dataType(), t2.dataType());
//	Tensor out;
//	NUMBER_TYPE_CASES(t1.dataType(), MatMulOp<T>::matmult(t2, t1, out));
//	return out;
//}
