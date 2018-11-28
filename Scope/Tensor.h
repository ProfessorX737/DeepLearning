#pragma once

#include <initializer_list>

#include "types.h"


class Tensor {
private:
	typedef std::initializer_list<int64> init_list;
public:
	Tensor(DataType dt, init_list dim_sizes) : dt_(dt) {}
private:
	void* data_;
	DataType dt_;
};
