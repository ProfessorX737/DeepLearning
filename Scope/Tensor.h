#pragma once

#include <initializer_list>
#include <vector>
#include <stdlib.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include <sstream>

#include "types.h"

template<typename T, size_t NDIMS = 1>
struct TTypes {
	typedef Eigen::TensorMap<
		Eigen::Tensor<T, NDIMS, Eigen::RowMajor, Eigen::Index>,
		Eigen::Aligned> Tensor;
	typedef Eigen::TensorMap<
		Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::Index>,
		Eigen::Aligned> Matrix;
	typedef Eigen::TensorMap<
		Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::Index>,
		Eigen::Aligned> Vec;
	typedef Eigen::TensorMap<
		Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, Eigen::Index>,
		Eigen::Aligned> Scalar;
};

class Tensor {
private:
	typedef std::initializer_list<Eigen::Index> dim_init_list;
	static const int ALIGNMENT = 64;
public:
	typedef Eigen::Index Index;
	Tensor(const std::vector<Index>& dims, DataType dt = DT_FLOAT);
	Tensor(const dim_init_list& dims, DataType dt = DT_FLOAT);

	template<typename T> 
	void init(const std::initializer_list<T>& init_list);

	~Tensor();

	int numDims() { return dims_.size(); }

	int dimSize(int index);

	template<typename T> 
	T* data() { return reinterpret_cast<T*>(data_); }

	std::string dimString() { 
		std::stringstream ss;
		ss << "[";
		for (int i = 0; i < numDims() - 1; i++)  ss << dims_[i] << ",";
		ss << dims_[numDims() - 1] << "]";
		return ss.str();
	}

	template<size_t NDIMS>
	Eigen::DSizes<Index, NDIMS> eigenDims();

	template<typename T, size_t NDIMS>
	typename TTypes<T, NDIMS>::Tensor tensor();

	template<typename T>
	typename TTypes<T>::Matrix matrix() { return tensor<T, 2>(); }

	template<typename T>
	typename TTypes<T>::Vec vec() { return tensor<T, 1>(); }

	template<typename T, size_t NDIMS>
	typename TTypes<T, NDIMS>::Tensor shaped(dim_init_list new_dims);

	template<typename T>
	typename TTypes<T>::Vec asVec() { return shaped<T, 1>({ static_cast<Index>(num_elements_) }); }

private:
	void* data_;
	DataType dt_;
	std::vector<Eigen::Index> dims_;
	size_t num_elements_;

	void* allocate(size_t num_bytes);
};

template<typename T> 
void Tensor::init(const std::initializer_list<T>& init_list) {
	CHECK_EQ(DataTypeToEnum<T>::v(), dt_);
	CHECK_EQ(init_list.size(), num_elements_);
	std::copy_n(init_list.begin(), init_list.size(), data<T>());
}

template<size_t NDIMS>
Eigen::DSizes<Eigen::Index, NDIMS> Tensor::eigenDims() {
	CHECK_GE(NDIMS, numDims()) << "Asking for " << NDIMS << " dims from " 
		<< numDims() << " tensor";
	Eigen::DSizes<Eigen::Index, NDIMS> edims;
	for (int i = 0; i < numDims(); i++) {
		edims[i] = dims_[i];
	}
	for (int i = numDims(); i < NDIMS; i++) {
		edims[i] = 1;
	}
	return edims;
}

template<typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::tensor() {
	CHECK_EQ(DataTypeToEnum<T>::v(), dt_) << "wrong type";
	return typename TTypes<T, NDIMS>::Tensor(data<T>(), eigenDims<NDIMS>());
}

template<typename T, size_t NDIMS>
typename TTypes<T,NDIMS>::Tensor Tensor::shaped(dim_init_list new_dims) {
	CHECK_EQ(new_dims.size(), NDIMS);
	int64 new_num_elements = 1;
	Eigen::DSizes<Eigen::Index, NDIMS> edims;
	int i = 0;
	for(Eigen::Index d : new_dims) {
		CHECK_GE(d, 0);
		new_num_elements *= d;
		edims[i++] = d;
	}
	CHECK_EQ(new_num_elements, num_elements_);
	return typename TTypes<T, NDIMS>::Tensor(data<T>(), edims);
}
