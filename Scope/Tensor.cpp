#include "Tensor.h"
#include <cstdlib>
#include <memory>
#include <algorithm>
#include "logging.h"
#include <malloc.h>
#include <stdio.h>

Tensor::Tensor() : buffer_(std::make_shared<TensorBuffer>()), dt_(DT_INVALID) {}

Tensor::Tensor(const TensorShape& shape, DataType dt) {
	init(shape, dt);
}

Tensor::Tensor(const dim_init_list& dims, DataType dt) {
	init(dims, dt);
}

void Tensor::init(const TensorShape& shape, DataType dt) {
	shape_ = std::move(shape);
	dt_ = dt;
	buffer_ = std::make_shared<TensorBuffer>();
	dt_size_ = DataTypeSize(dt_);
	buffer_->allocate(numElements()*dt_size_);
}

void Tensor::init(const dim_init_list& dims, DataType dt) {
	shape_.init(dims);
	dt_ = dt;
	buffer_ = std::make_shared<TensorBuffer>();
	dt_size_ = DataTypeSize(dt_);
	buffer_->allocate(numElements()*dt_size_);
}

void Tensor::sharedCopyInit(const Tensor& other) {
	sharedCopyInit(other, other.shape_);
}

void Tensor::sharedCopyInit(const Tensor& other, const TensorShape& shape) {
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
}

void Tensor::TensorBuffer::free()
{
	if (data == nullptr) return;
#if defined(_WIN32) || defined(WIN32)
	_aligned_free(data);
#else
	free(data);
#endif
}
