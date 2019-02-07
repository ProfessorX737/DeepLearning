#ifndef TENSOR_H
#define TENSOR_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include "TensorShape.h"
#include "logging.h"
#include <iostream>
#include "types.h"


template<typename T, size_t NDIMS = 1>
struct TTypes {
    typedef Eigen::Tensor<T, NDIMS, Eigen::RowMajor, Eigen::Index> Tensor;
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vec;
    typedef Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, Eigen::Index> Scalar;
};

template<typename T, size_t NDIMS = 1>
struct MTTypes {
    typedef Eigen::TensorMap<typename TTypes<T,NDIMS>::Tensor, Eigen::Aligned> Tensor;
    typedef Eigen::Map<typename TTypes<T,NDIMS>::Matrix, Eigen::Aligned> Matrix;
    typedef Eigen::Map<typename TTypes<T,NDIMS>::Vec, Eigen::Aligned> Vec;
    typedef Eigen::TensorMap<typename TTypes<T,NDIMS>::Scalar, Eigen::Aligned> Scalar;
};

// TODO: optimize for scalars
// TODO(adv): modern c++ small-object allocation
class Tensor {
public:
#ifdef CUSTOM_MAX_DIMS
    static constexpr int MAX_DIMS = CUSTOM_MAX_DIMS;
#else
    static constexpr int MAX_DIMS = 3;
#endif
	static const int ALIGNMENT = 64;
	typedef TensorShape::dim_init_list dim_init_list;
	typedef TensorShape::Dim Dim;
	Tensor();
	Tensor(const TensorShape& shape, DataType dt = DT_FLOAT);
	Tensor(const dim_init_list& dims, DataType dt = DT_FLOAT);
	void init(const TensorShape& shape, DataType dt = DT_FLOAT);
	void init(const dim_init_list& dims, DataType dt = DT_FLOAT);
    
    // if allocation space is less than or equal to pre-allocated space
    // the pre-allocated space is reused
    void recycle(const TensorShape& shape, DataType dt);
    
	void sharedCopyInit(const Tensor& other);
	void sharedCopyInit(const Tensor& other, const TensorShape& shape);
    template<typename T>
    void sharedCopyInit(const Tensor& other, int batchIndex);

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
    bool hasSameShape(const Tensor& other) const { return shape_.isSameShape(other.shape()); }

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

	template<typename T, typename S>
	Tensor scalarMult(const S scalar) const;

	template<typename T>
	Tensor cWiseMult(const Tensor& other) const;

	template<size_t NDIMS>
	Eigen::DSizes<Dim, NDIMS> eigenDimsPadLeft() const;
    
	template<size_t NDIMS>
	Eigen::DSizes<Dim, NDIMS> eigenDimsPadRight() const;
    
    template<size_t NDIMS>
    static Eigen::DSizes<Dim,NDIMS> eigenDimsPadLeft(Eigen::DSizes<Dim,NDIMS> rawDims);
    
    template<size_t FROM_DIMS, size_t TO_DIMS>
    static Eigen::DSizes<Eigen::Index,TO_DIMS> eigenDimsPadRight(Eigen::DSizes<Eigen::Index,FROM_DIMS> rawDims);
        
    template<size_t FROM_DIMS, size_t TO_DIMS>
    static Eigen::DSizes<Eigen::Index,TO_DIMS> eigenDimsPadLeft(Eigen::DSizes<Eigen::Index,FROM_DIMS> rawDims);

	template<typename T, size_t NDIMS>
	typename MTTypes<T, NDIMS>::Tensor tensor() const;
    
	template<typename T, size_t NDIMS>
	typename MTTypes<T, NDIMS>::Tensor tensorPadRight() const;
    
	template<typename T, size_t NDIMS>
	typename MTTypes<T, NDIMS>::Tensor tensorPadLeft() const;
    
	template<typename T, size_t FROM_DIMS, size_t TO_DIMS>
    static typename MTTypes<T, TO_DIMS>::Tensor tensorPadRight(typename TTypes<T,FROM_DIMS>::Tensor& in);
    
	template<typename T, size_t FROM_DIMS, size_t TO_DIMS>
    static typename MTTypes<T, TO_DIMS>::Tensor tensorPadLeft(typename TTypes<T,FROM_DIMS>::Tensor& in);

	template<typename T>
	typename MTTypes<T>::Matrix matrix() const;

	template<typename T>
	typename MTTypes<T>::Matrix asMatrix(Dim rows, Dim cols) const;

	template<typename T>
	typename MTTypes<T>::Vec asVec() const;

	template<typename T, size_t NDIMS>
	typename MTTypes<T, NDIMS>::Tensor shaped(const dim_init_list& new_dims) const;


private:
	class TensorBuffer {
	public:
		TensorBuffer() : data(nullptr) {}
		~TensorBuffer() {
			free();
		}
		void allocate(size_t num_bytes);
		void* data;
        bool ownsData = false;
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
Eigen::DSizes<Eigen::Index, NDIMS> Tensor::eigenDimsPadLeft() const {
	CHECK_GE(NDIMS, numDims()) << "Asking for " << NDIMS
		<< " dim tensor of shape " << dimString();
	Eigen::DSizes<Eigen::Index, NDIMS> edims;
    int paddingEnd = NDIMS - numDims();
    int dimIndex = 0;
	for (int i = paddingEnd; i < NDIMS; i++) {
		edims[i] = dimSize(dimIndex);
        dimIndex++;
	}
    for(int i = 0; i < paddingEnd; i++) {
        edims[i] = 1;
    }
	return edims;
}

template<size_t NDIMS>
Eigen::DSizes<Eigen::Index, NDIMS> Tensor::eigenDimsPadRight() const {
	CHECK_GE(NDIMS, numDims()) << "Asking for " << NDIMS
		<< " dim tensor of shape " << dimString();
	Eigen::DSizes<Eigen::Index, NDIMS> edims;
	for (int i = 0; i < numDims(); i++) {
		edims[i] = dimSize(i);
	}
    for(int i = numDims(); i < NDIMS; i++) {
        edims[i] = 1;
    }
	return edims;
}

template<size_t FROM_DIMS, size_t TO_DIMS>
Eigen::DSizes<Eigen::Index,TO_DIMS> Tensor::eigenDimsPadLeft(Eigen::DSizes<Eigen::Index,FROM_DIMS> rawDims) {
    CHECK_GT(TO_DIMS,FROM_DIMS) << " Asking for " << TO_DIMS
        << " dim tensor from " << FROM_DIMS << " dim tensor";
	Eigen::DSizes<Eigen::Index, TO_DIMS> paddedDims;
    int paddingEnd = TO_DIMS - FROM_DIMS;
    int dimIndex = 0;
	for (int i = paddingEnd; i < TO_DIMS; i++) {
		paddedDims[i] = rawDims[dimIndex];
        dimIndex++;
	}
    for(int i = 0; i < paddingEnd; i++) {
        paddedDims[i] = 1;
    }
	return paddedDims;
}

template<size_t FROM_DIMS, size_t TO_DIMS>
Eigen::DSizes<Eigen::Index,TO_DIMS> Tensor::eigenDimsPadRight(Eigen::DSizes<Eigen::Index,FROM_DIMS> rawDims) {
    CHECK_GT(TO_DIMS,FROM_DIMS) << " Asking for " << TO_DIMS
        << " dim tensor from " << FROM_DIMS << " dim tensor";
    Eigen::DSizes<Eigen::Index,TO_DIMS> paddedDims;
    for(int i = 0; i < FROM_DIMS; i++) {
        paddedDims[i] = rawDims[i];
    }
    for(int i = rawDims.size(); i < TO_DIMS; i++) {
        paddedDims[i] = 1;
    }
    return paddedDims;
}

template<typename T, size_t NDIMS>
typename MTTypes<T, NDIMS>::Tensor Tensor::tensor() const {
    return tensorPadLeft<T,NDIMS>();
}

template<typename T, size_t NDIMS>
typename MTTypes<T, NDIMS>::Tensor Tensor::tensorPadRight() const {
	CHECK_EQ(DataTypeToEnum<T>::v(), dt_) << "wrong type";
	return typename MTTypes<T, NDIMS>::Tensor(data<T>(), eigenDimsPadRight<NDIMS>());
}

template<typename T, size_t NDIMS>
typename MTTypes<T, NDIMS>::Tensor Tensor::tensorPadLeft() const {
	CHECK_EQ(DataTypeToEnum<T>::v(), dt_) << "wrong type";
	return typename MTTypes<T, NDIMS>::Tensor(data<T>(), eigenDimsPadLeft<NDIMS>());
}

template<typename T, size_t FROM_DIMS, size_t TO_DIMS>
typename MTTypes<T, TO_DIMS>::Tensor Tensor::tensorPadRight(typename TTypes<T,FROM_DIMS>::Tensor& in) {
    CHECK_GT(TO_DIMS,FROM_DIMS) << "num output dims should be greater than num intput dims";
    return typename MTTypes<T,TO_DIMS>::Tensor(in.data(),eigenDimsPadRight<FROM_DIMS,TO_DIMS>(in.dimensions()));
}

template<typename T, size_t FROM_DIMS, size_t TO_DIMS>
typename MTTypes<T, TO_DIMS>::Tensor Tensor::tensorPadLeft(typename TTypes<T,FROM_DIMS>::Tensor& in) {
    CHECK_GT(TO_DIMS,FROM_DIMS) << "num output dims should be greater than num intput dims";
    return typename MTTypes<T,TO_DIMS>::Tensor(in.data(),eigenDimsPadLeft<FROM_DIMS,TO_DIMS>(in.dimensions()));
}

template<typename T>
typename MTTypes<T>::Matrix Tensor::matrix() const { 
	CHECK_EQ(numDims(), 2);
	CHECK_EQ(DataTypeToEnum<T>::v(), dt_);
	return typename MTTypes<T>::Matrix(data<T>(),
		dimSize(0),dimSize(1)); 
}

template<typename T>
typename MTTypes<T>::Matrix Tensor::asMatrix(Dim rows, Dim cols) const { 
	CHECK_EQ(rows*cols,numElements());
	CHECK_EQ(DataTypeToEnum<T>::v(), dt_);
	return typename MTTypes<T>::Matrix(data<T>(),rows,cols); 
}

template<typename T>
typename MTTypes<T>::Vec Tensor::asVec() const { 
	CHECK_GE(numDims(), 1);
	return typename MTTypes<T>::Vec(data<T>(),numElements(),1);
}

template<typename T, size_t NDIMS>
typename MTTypes<T, NDIMS>::Tensor Tensor::shaped(const dim_init_list& new_dims) const {
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
	return typename MTTypes<T, NDIMS>::Tensor(data<T>(), edims);
}

template<typename T> 
Tensor operator*(const T scalar, const Tensor& t) {
	return t * scalar;
}

template<typename T>
void Tensor::sharedCopyInit(const Tensor& other, int batchIndex) {
    if((batchIndex < 0) || (other.numDims() < 2)) {
        sharedCopyInit(other);
        return;
    }
    DCHECK_LT(batchIndex, other.dimSize(0));
    dt_ = other.dataType();
    int batchSize = 1;
    shape_.addDim(1);
    for(int i = 1; i < other.numDims(); i++) {
        batchSize *= other.dimSize(i);
        shape_.addDim(other.dimSize(i));
    }
    DCHECK_EQ(buffer_->ownsData, false);
    buffer_->data = other.data<T>() + (batchIndex * batchSize);
}

#include "Multiply.h"

// multiplies tensor by a scalar element-wise
// expensive as it allocates a new tensor on the heap
template<typename T, typename S>
Tensor Tensor::scalarMult(const S scalar) const {
	Tensor res;
	ScalarMultiplyOp<T>::scalarMultiply(*this, static_cast<T>(scalar), res);
	return res;
}

// multiplies two tensors together element-wise
// expensive as it allocates a new tensor on the heap
template<typename T>
Tensor Tensor::cWiseMult(const Tensor& other) const {
	Tensor res;
	CWiseMultiplyOp<T>::cWiseMultiply(*this, other, res);
	return res;
}

#endif // TENSOR_H