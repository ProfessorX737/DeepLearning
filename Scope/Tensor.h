#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
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

// TODO: optimize for scalars
// TODO(adv): modern c++ small-object allocation
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
	int dataTypeSize() { return dt_size_; }

	template<typename T> 
	T* data() const { return reinterpret_cast<T*>(buffer_->data); }

	Tensor& operator=(const Tensor& other) {
		sharedCopyInit(other);
		return *this;
	}

	template<typename T> 
	Tensor operator*(const T scalar) const {
        CHECK_EQ(dt_,DataTypeToEnum<T>::v());
		Tensor res(shape(), dt_);
		res.asVec<T>() = (asVec<T>().array() * scalar).matrix();
        return res;
	}

	template<typename T>
	void multiply(const Tensor& t, bool transA = false, bool transB = false);

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
	class TensorBuffer {
	public:
		TensorBuffer() : data(nullptr) {}
		~TensorBuffer() {
			free();
		}
		void allocate(size_t num_bytes);
		void* data;
	private:
		void free();
	};
	std::shared_ptr<TensorBuffer> buffer_;
	DataType dt_;
	int dt_size_;
	TensorShape shape_;
};

template<typename T> 
void Tensor::fill(const std::initializer_list<T>& init_list) {
	CHECK_EQ(DataTypeToEnum<T>::v(), dt_);
	CHECK_EQ(init_list.size(), numElements());
	std::copy_n(init_list.begin(), init_list.size(), data<T>());
}

template<typename T> 
void Tensor::init(const std::initializer_list<T>& init_list) {
    init(TensorShape({ static_cast<Eigen::Index>(init_list.size()),1 }), DataTypeToEnum<T>::v());
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
	//CHECK_EQ(DataTypeToEnum<T>::v(), dt_);
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

template<typename T> 
Tensor operator*(const T scalar, const Tensor& t) {
	return t * scalar;
}

#include "MatMul.h"

template<typename T>
void Tensor::multiply(const Tensor& t, bool transA, bool transB) {
	Tensor thisCopy;
	thisCopy = *this;
	MatMulOp<T>::mult(thisCopy, t, *this, transA, transB);
}
