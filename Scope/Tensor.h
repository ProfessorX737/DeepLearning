#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include "TensorShape.h"
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
public:
	static const int ALIGNMENT = 64;
	typedef TensorShape::dim_init_list dim_init_list;
	typedef TensorShape::Dim Dim;

	Tensor();
	Tensor(const TensorShape& shape, DataType dt = DT_FLOAT);
	Tensor(const dim_init_list& dims, DataType dt = DT_FLOAT);
	void init(const TensorShape& shape, DataType dt = DT_FLOAT);
	void init(const dim_init_list& dims, DataType dt = DT_FLOAT);
	void sharedCopyInit(const Tensor& other);
	void sharedCopyInit(const Tensor& other, const TensorShape& shape);

	template<typename T> 
	void fill(const std::initializer_list<T>& init_list);
	template<typename T> 
	void init(const std::initializer_list<T>& init_list);

	~Tensor();

	int numDims() const { return shape_.numDims(); }
	int dimSize(int i) const { return shape_.dimSize(i); }
	int numElements() const { return shape_.numElements(); }
	DataType dataType() const { return dt_; }

	template<typename T> 
	T* data() { return reinterpret_cast<T*>(data_); }

	std::string dimString() const { return shape_.dimString(); }

	template<size_t NDIMS>
	Eigen::DSizes<Dim, NDIMS> eigenDims();

	template<typename T, size_t NDIMS>
	typename TTypes<T, NDIMS>::Tensor tensor();

	template<typename T>
	typename TTypes<T>::Matrix matrix() { return tensor<T, 2>(); }

	template<typename T>
	typename TTypes<T>::Vec vec() { return tensor<T, 1>(); }

	template<typename T, size_t NDIMS>
	typename TTypes<T, NDIMS>::Tensor shaped(dim_init_list new_dims);

	template<typename T>
	typename TTypes<T>::Vec asVec() { return shaped<T, 1>({ static_cast<Dim>(numElements())}); }

private:
	void* data_;
	DataType dt_;
	TensorShape shape_;
	bool owns_data_;
	void* allocate(size_t num_bytes);
};

template<typename T> 
void Tensor::fill(const std::initializer_list<T>& init_list) {
	CHECK_EQ(DataTypeToEnum<T>::v(), dt_);
	CHECK_EQ(init_list.size(), numElements());
	std::copy_n(init_list.begin(), init_list.size(), data<T>());
}

template<typename T> 
void Tensor::init(const std::initializer_list<T>& init_list) {
	init(TensorShape({ static_cast<Eigen::Index>(init_list.size()) }), DataTypeToEnum<T>::v());
	//std::cout << numElements() << std::endl;
	std::copy_n(init_list.begin(), init_list.size(), data<T>());
}

template<size_t NDIMS>
Eigen::DSizes<Eigen::Index, NDIMS> Tensor::eigenDims() {
	CHECK_GE(NDIMS, numDims()) << "Asking for " << NDIMS << " dims from " 
		<< numDims() << " tensor";
	Eigen::DSizes<Eigen::Dim, NDIMS> edims;
	for (int i = 0; i < numDims(); i++) {
		edims[i] = dimSize(i);
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
	for(Dim d : new_dims) {
		CHECK_GE(d, 0);
		new_num_elements *= d;
		edims[i++] = d;
	}
	CHECK_EQ(new_num_elements, numElements());
	return typename TTypes<T, NDIMS>::Tensor(data<T>(), edims);
}
