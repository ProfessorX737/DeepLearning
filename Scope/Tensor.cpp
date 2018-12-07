#include "Tensor.h"
#include <cstdlib>
#include <memory>
#include <algorithm>
#include "logging.h"
#include <malloc.h>
#include <stdio.h>
#include <iostream>

Tensor::Tensor() : data_(nullptr), dt_(DT_INVALID) {}

Tensor::Tensor(const TensorShape& shape, DataType dt) {
	init(shape, dt);
}

Tensor::Tensor(const dim_init_list& dims, DataType dt) {
	init(dims, dt);
}

void Tensor::init(const TensorShape& shape, DataType dt) {
	if (data_) delete data_;
	shape_ = std::move(shape);
	dt_ = dt;
	data_ = allocate(numElements()*DataTypeSize(dt_));
	owns_data_ = true;
}

void Tensor::init(const dim_init_list& dims, DataType dt) {
	if (data_) delete data_;
	shape_.init(dims);
	dt_ = dt;
	owns_data_ = true;
	data_ = allocate(numElements()*DataTypeSize(dt_));
}

void Tensor::sharedCopyInit(const Tensor& other) {
	sharedCopyInit(other, other.shape_);
}

// copy another tensor and shares its data ptr
// but does not manage its deletion
void Tensor::sharedCopyInit(const Tensor& other, const TensorShape& shape) {
	CHECK_EQ(shape.numElements(), other.numElements());
	dt_ = other.dataType();
	if (data_ != other.data_) {
		if (data_) delete data_;
		data_ = other.data_;
	}
	shape_ = std::move(shape);
	owns_data_ = false;
}

Tensor::~Tensor() {
	if (owns_data_) {
		if (data_) delete data_;
	}
}

void* Tensor::allocate(size_t num_bytes) {
	if (num_bytes <= 0) return nullptr;
	void* ptr = nullptr;
#if defined(_WIN32) || defined(WIN32)
	ptr = _aligned_malloc(num_bytes, ALIGNMENT);
#else
	DCHECK_EQ(ALIGNMENT % sizeof(void*), 0);
	DCHECK_POW2(ALIGNMENT);
	CHECK_GE(ALIGNMENT, sizeof(void*));
	if (posix_memalign(&ptr, ALIGNMENT, num_bytes)) ptr = nullptr;
#endif
	CHECK_NOT_NULL(ptr) << "Failed to allocate " << num_bytes << " bytes";
	return ptr;
}
