#include "TensorShape.h"
#include "logging.h"
#include <sstream>
#include <iostream>

TensorShape::TensorShape(const dim_init_list& dims) : num_elements_(0) {
	init(dims);
}

void TensorShape::init(const dim_init_list& dims) {
    if(dims.size() == 0) return;
	num_elements_ = 1;
	for (Dim d : dims) {
		addDim(d);
	}
}

void TensorShape::addDim(Dim d) {
	DCHECK_GT(d, 0);
	num_elements_ *= d;
	dims_.push_back(d);
}

int TensorShape::dimSize(int i) const {
	DCHECK_GE(i, 0);
	DCHECK_LT(i, numDims());
	return static_cast<int>(dims_[i]);
}

int TensorShape::numElements() const {
    if(dims_.size() == 0) return 0;
    int nelements = 1;
    for(Dim d : dims_) {
        nelements *= d;
    }
    return nelements;
}

std::string TensorShape::dimString() const { 
	std::stringstream ss;
	ss << "[";
	for (int i = 0; i < numDims() - 1; i++)  ss << dims_[i] << ",";
	ss << dims_[numDims() - 1] << "]";
	return ss.str();
}

bool TensorShape::isSameShape(const TensorShape& other) const {
	if (numDims() != other.numDims()) return false;
	for (int i = 0; i < numDims(); i++) {
		if (dimSize(i) != other.dimSize(i)) return false;
	}
	return true;
}
