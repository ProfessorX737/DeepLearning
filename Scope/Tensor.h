#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include "TensorShape.h"
#include "types.h"

template<typename T, size_t NDIMS = 1>
struct TTypes {
	typedef Eigen::TensorMap<
		Eigen::Tensor<T, NDIMS, Eigen::RowMajor, Eigen::Index>,
		Eigen::Aligned> Tensor;
	typedef Eigen::Map<
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
		Eigen::Aligned> Matrix;
	typedef Eigen::Map<
		Eigen::Matrix<T, Eigen::Dynamic, 1>,
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
	std::string dimString() const { return shape_.dimString(); }
	TensorShape shape() const { return shape_; }

	template<typename T> 
	T* data() const { return reinterpret_cast<T*>(data_); }


	template<size_t NDIMS>
	Eigen::DSizes<Dim, NDIMS> eigenDims() const;

	template<typename T, size_t NDIMS>
	typename TTypes<T, NDIMS>::Tensor tensor() const;

	template<typename T>
	typename TTypes<T>::Matrix matrix() const;

	template<typename T>
	typename TTypes<T>::Matrix asMatrix(Dim rows, Dim cols) const;

	template<typename T>
	typename TTypes<T>::Vec asVec() const;

	template<typename T, size_t NDIMS>
	typename TTypes<T, NDIMS>::Tensor shaped(const dim_init_list& new_dims) const;

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
	std::copy_n(init_list.begin(), init_list.size(), data<T>());
}

template<size_t NDIMS>
Eigen::DSizes<Eigen::Index, NDIMS> Tensor::eigenDims() const {
	CHECK_EQ(NDIMS, numDims()) << "Asking for " << NDIMS
		<< " dim tensor of shape " << dimString();
	Eigen::DSizes<Eigen::Index, NDIMS> edims;
	for (int i = 0; i < numDims(); i++) {
		edims[i] = dimSize(i);
	}
	return edims;
}

template<typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::tensor() const {
	CHECK_EQ(DataTypeToEnum<T>::v(), dt_) << "wrong type";
	return typename TTypes<T, NDIMS>::Tensor(data<T>(), eigenDims<NDIMS>());
}

template<typename T>
typename TTypes<T>::Matrix Tensor::matrix() const { 
	CHECK_EQ(numDims(), 2);
	CHECK_EQ(DataTypeToEnum<T>::v(), dt_);
	return typename TTypes<T>::Matrix(data<T>(),
		dimSize(0),dimSize(1)); 
}

template<typename T>
typename TTypes<T>::Matrix Tensor::asMatrix(Dim rows, Dim cols) const { 
	CHECK_EQ(rows*cols,numElements());
	CHECK_EQ(DataTypeToEnum<T>::v(), dt_);
	return typename TTypes<T>::Matrix(data<T>(),rows,cols); 
}

template<typename T>
typename TTypes<T>::Vec Tensor::asVec() const { 
	CHECK_GE(numDims(), 1);
	CHECK_EQ(DataTypeToEnum<T>::v(), dt_);
	return typename TTypes<T>::Vec(data<T>(),numElements(),1); 
}

template<typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::shaped(const dim_init_list& new_dims) const {
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
