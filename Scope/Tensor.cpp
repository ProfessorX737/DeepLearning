#include "Tensor.h"
#include <cstdlib>
#include <stdlib.h>
#include <memory>
#include <algorithm>
#include <stdio.h>
#if defined(_WIN32) || defined(WIN32)
#include <malloc.h>
#endif

Tensor::Tensor() : buffer_(std::make_shared<TensorBuffer>()), dt_(DT_INVALID) {}

Tensor::Tensor(const TensorShape& shape, DataType dt) {
	init(shape, dt);
}

Tensor::Tensor(const dim_init_list& dims, DataType dt) {
	init(dims, dt);
}

void Tensor::init(const TensorShape& shape, DataType dt) {
    if(!shape_.isSameShape(shape) || dt_ != dt) {
    	dt_ = dt;
        dt_size_ = DataTypeSize::v(dt_);
    	buffer_ = std::make_shared<TensorBuffer>();
    	buffer_->allocate(shape.numElements()*dt_size_);
        //std::cout << "calling expensive tensor allocate" << std::endl;
    	shape_ = std::move(shape);
    }
}

void Tensor::recycle(const TensorShape& shape, DataType dt) {
    if(!shape_.isSameShape(shape) || dt_ != dt) {
    	dt_ = dt;
        dt_size_ = DataTypeSize::v(dt_);
        if(numElements() < shape.numElements()) {
        	buffer_ = std::make_shared<TensorBuffer>();
        	buffer_->allocate(shape.numElements()*dt_size_);
        }
    	shape_ = std::move(shape);
    }
}

void Tensor::init(const dim_init_list& dims, DataType dt) {
    TensorShape shape(dims);
    init(shape,dt);
}

void Tensor::sharedCopyInit(const Tensor& other) {
	sharedCopyInit(other, other.shape_);
}

void Tensor::sharedCopyInit(const Tensor& other, const TensorShape& shape) {
    //std::cout << shape.numElements() << " " << other.numElements() << std::endl;
	CHECK_EQ(shape.numElements(), other.numElements());
	dt_ = other.dataType();
	buffer_ = other.buffer_;
	shape_ = std::move(shape);
}

Tensor::~Tensor() {}

void Tensor::TensorBuffer::allocate(size_t num_bytes) {
	if (num_bytes <= 0) {
		data = nullptr;
		return;
	}
	data = nullptr;
#if defined(_WIN32) || defined(WIN32)
	data = _aligned_malloc(num_bytes, ALIGNMENT);
#else
	DCHECK_EQ(ALIGNMENT % sizeof(void*), 0);
	DCHECK_POW2(ALIGNMENT);
	CHECK_GE(ALIGNMENT, sizeof(void*));
	if (posix_memalign(&data, ALIGNMENT, num_bytes)) data = nullptr;
#endif
	CHECK_NOT_NULL(data) << "Failed to allocate " << num_bytes << " bytes";
    ownsData = true;
}

void Tensor::TensorBuffer::free()
{
	if (data == nullptr || ownsData == false) return;
#if defined(_WIN32) || defined(WIN32)
	_aligned_free(data);
#else
	::free(data);
#endif
}
