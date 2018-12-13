#pragma once
#include "BinaryOp.h"

template<typename T>
class MatMul : public BinaryOp {
public:
	MatMul(Graph& graph, const Node& a, const Node& b) 
		: BinaryOp(graph, a, b, "MatMul"), transA_(0), transB_(0) {}

	MatMul(Graph& graph, const Node& a, const Node& b, bool transA, bool transB)
		: BinaryOp(graph, a, b, "MatMul"), transA_(transA), transB_(transB) {}

private:
	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using ConstMatrixMap = Eigen::Map<const Matrix>;
	using MatrixMap = Eigen::Map<Matrix>;

	ConstMatrixMap constEigenMatrixSlice(const Tensor& t, int slice) const {
		return ConstMatrixMap(t.data<T>() + slice * t.dimSize(1) * t.dimSize(2),
			t.dimSize(1), t.dimSize(2));
	}

	MatrixMap eigenMatrixSlice(Tensor& t, int slice) const {
		return MatrixMap(t.data<T>() + slice * t.dimSize(1) * t.dimSize(2),
			t.dimSize(1), t.dimSize(2));
	}

	void binaryOp(const Tensor& a, const Tensor& b, Tensor& out) const override {
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
		TensorShape::Dim nbatch = out_shape.numElements();

		// make reshaped a & b that have most 3 dims
		Tensor rA, rB, rOut;
		rA.sharedCopyInit(a, TensorShape({ nbatch,arows,acols }));
		rB.sharedCopyInit(b, TensorShape({ nbatch,brows,bcols }));

		if (transA_) std::swap(arows, acols);
		if (transB_) std::swap(bcols, brows);

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

			if (!transA_) {
				if (!transB_) {
					matOut.noalias() = matA * matB;
				}
				else {
					matOut.noalias() = matA * matB.transpose();
				}
			}
			else {
				if (!transB_) {
					matOut.noalias() = matA.transpose() * matB;
				}
				else {
					matOut.noalias() = matA.transpose() * matB.transpose();
				}
			}
		}
	}

	bool transA_;
	bool transB_;
};
