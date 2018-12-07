#include "Variable.h"

Variable::Variable(const Tensor::dim_init_list& shape, DataType dt) : Node("Variable") {
	t_.init(shape, dt);
}
Variable::Variable(const TensorShape& shape, DataType dt) : Node("Variable") {
	t_.init(shape, dt);
}

Tensor Variable::tensor() {
	Tensor t;
	t.sharedCopyInit(t_);
	return t;
}
