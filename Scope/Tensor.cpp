#include "Tensor.h"
#include <cstdlib>
#include <memory>
#include <algorithm>
#include "logging.h"
#include <malloc.h>
#include <stdio.h>
#include <iostream>

Tensor::Tensor(const std::vector<Index>& dims, DataType dt)
	: dims_(std::move(dims)), dt_(dt), num_elements_(1) {
	std::cout << "calling pass vector constructor" << std::endl;
	CHECK_GT(dims_.size(), 0);
	for (Eigen::Index s : dims_) {
		CHECK_GE(s, 0);
		num_elements_ *= s;
	}
	data_ = allocate(num_elements_*DataTypeSize(dt_));
}

Tensor::Tensor(const dim_init_list& dims, DataType dt) 
	: dims_(std::move(dims)), dt_(dt), num_elements_(1) {
	CHECK_GT(dims_.size(), 0);
	for (Eigen::Index s : dims_) {
		CHECK_GE(s, 0);
		num_elements_ *= s;
	}
	data_ = allocate(num_elements_*DataTypeSize(dt_));
	std::vector<Index> dimz = std::move(dims);
}

Tensor::~Tensor() {
	if (data_) delete data_;
}

int Tensor::dimSize(int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, numDims());
	return dims_[index];
}

void* Tensor::allocate(size_t num_bytes) {
	CHECK_GT(num_bytes, 0);
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
