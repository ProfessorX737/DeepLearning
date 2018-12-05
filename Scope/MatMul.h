#pragma once

#include "BinaryOp.h"
#include "Tensor.h"
#include "logging.h"

template<typename T>
class MatMul : public BinaryOp {
public:
	MatMul(Node* left, Node* right) : BinaryOp(left, right, "MatMul") {}
private:
	Tensor binaryOp(const Tensor& left, const Tensor& right) override {
		CHECK_EQ(left.numDims(), right.numDims()) 
			<< "left and right operands have different ndims: " 
			<< left.dimString() << " vs " << right.dimString();

		const int ndims = left.numDims();
		CHECK_GE(ndims,2) 
			<< "left and right operands need to have >= 2 dims";

		CHECK_EQ(left.dimSize(left.numDims - 1), right.dimSize(0)) 
			<< "Inner dims must equal: " << left.dimString() << "*" 
			<< right.dimString();

		std::vector<Tensor::Index> out_dims;
		for (int i = 0; i < ndims - 2; i++) {
			CHECK_EQ(left.dimSize(i), right.dimSize(i))
				<< "dimSize(" << i << ") of" << left.dimString() << " and " 
				<< right.dimString() << " must be the same";
			out_dims.push_back(left.dimSize(i));
		}

		return Tensor({ 1,2 });
	}
};
