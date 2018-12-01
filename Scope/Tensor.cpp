#include "Tensor.h"
#include <cstdlib>

Tensor::Tensor(DataType dt, init_list dim_sizes) : dt_(dt) {
	for (int64 s : dim_sizes) {
		dims_.push_back(s);
	}
}

template<typename T>
Tensor::Tensor(const std::initializer_list<T>& v) {
	dims_.push_back(1);
	dt_ = DataTypeToEnum<T>::v();
	num_elements_ = v.size();
	data_ = ::operator new(sizeof(T)*num_elements);
	std::copy_n(v.begin(), v.size(), static_cast<T*>(data_));
}

template<typename T>
void* Tensor::Allocate(int num_elements) {
	DCHECK_NE(num_elements, 0);
	uint nbytes = sizeof(T)*num_elements;
	void* ptr = std::aligned_alloc(ALIGNMENT, nbytes);
	CHECK_NOT_NULL(ptr) << "Failed to allocate " << nbytes << " bytes";
	return ptr;
}

Tensor::~Tensor() {
	//TODO: implement reference counted buffer class so 
	// multiple tensors can share the same data
	if (data_) delete data_;
}
