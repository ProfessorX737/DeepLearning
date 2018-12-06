#pragma once
#include "Node.h"
#include "Tensor.h"

class Variable : public Node {
public:
	Variable(const Tensor::dim_init_list& shape, DataType dt);
	Variable(const TensorShape& shape, DataType dt);
	template<typename T>
	void fill(const std::initializer_list<T>& list) { t_.fill<T>(list); }
	void eval(Tensor& out) { out.sharedCopyInit(t_); };
	Tensor t_;
};
