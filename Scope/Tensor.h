#pragma once

#include <initializer_list>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <malloc.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

#include "types.h"

template<typename T, int NDIMS = 1>
struct TTypes {
	typedef Eigen::TensorMap<
		Eigen::Tensor<T, NDIMS, Eigen::RowMajor, Eigen::Index>,
		Eigen::Aligned> Tensor;
	typedef Eigen::TensorMap<
		Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, Eigen::Index>,
		Eigen::Aligned> ConstTensor;
	typedef Eigen::TensorMap<
		Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::Index>,
		Eigen::Aligned> Vec;
	typedef Eigen::TensorMap<
		Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::Index>,
		Eigen::Aligned> ConstVec;
	typedef Eigen::TensorMap<
		Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, Eigen::Index>,
		Eigen::Aligned> Scalar;
	typedef Eigen::TensorMap<
		Eigen::TensorFixedSize<const T, Eigen::Sizes<>, Eigen::RowMajor, Eigen::Index>,
		Eigen::Aligned> ConstScalar;
};

class Tensor {
private:
	static const int ALIGNMENT = 64;
	typedef std::initializer_list<int64> init_list;
public:
	Tensor(DataType dt, init_list dim_sizes);

	// initialize 1D tensor with v.size() columns
	template<typename T>
	Tensor(const std::initializer_list<T>& v);

	int num_dims() { return dims_.size(); }

	template<typename T> 
	T* data() { return reinterpret_cast<T>(data_); }

	template<typename T>
	void* Allocate(int num_elements);

	~Tensor();
private:
	void* data_;
	DataType dt_;
	std::vector<int64> dims_;
	int num_elements_;
};
