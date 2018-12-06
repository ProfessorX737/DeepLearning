#pragma once
#include "BinaryOp.h"

template<typename T>
class MatMul : public BinaryOp {
public:
	MatMul(Node* a, Node* b) 
		: BinaryOp(a, b, "MatMul"), transA_(0), transB_(0) {}

	MatMul(Node* a, Node* b, bool transA, bool transB)
		: BinaryOp(a, b, "MatMul"), transA_(transA), transB_(transB) {}

private:
	using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using MatrixMap = Eigen::Map<const Matrix>;

	MatrixMap eigenMatrixSlice(const Tensor& t, int slice) {
		return MatrixMap(t.asVec<T>().data() + slice * t.dimSize(1) * t.dimSize(2),
			t.dimSize(1), t.dimSize(2));
	}

	void binaryOp(Tensor& a, Tensor& b, Tensor* out) override {
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

		if (transA_) std::swap(arows, acols);
		if (transB_) std::swap(acols, arows);

		CHECK_EQ(acols, brows) 
			<< "Inner dims must equal: " << a.dimString() << " vs " 
			<< b.dimString();

		// size of the batch dim is equal to product of all dim sizes except last two
		TensorShape::Dim nbatch = out_shape.numElements();

		out_shape.addDim(arows);
		out_shape.addDim(bcols);

		out->init(out_shape, a.dataType());

		// make reshaped a & b that have most 3 dims
		Tensor rA, rB;
		rA.sharedCopyInit(a, TensorShape({ nbatch,arows,acols }));
		rB.sharedCopyInit(b, TensorShape({ nbatch,brows,bcols });

		for (int i = 0; i < nbatch; i++) {
			auto matA = eigenMatrixSlice(rA, i);
			auto matB = eigenMatrixSlice(rB, i);
			auto matOut = eigenMatrixSlice(out, i);

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
