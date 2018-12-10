#include "Variable.h"

Variable::Variable(Graph& graph, const Tensor::dim_init_list& shape, DataType dt) : Node("Variable", graph) {
	t_.init(shape, dt);
}
Variable::Variable(Graph& graph, const TensorShape& shape, DataType dt) : Node("Variable", graph) {
	t_.init(shape, dt);
}

Tensor Variable::tensor() {
	Tensor t;
	t.sharedCopyInit(t_);
	return t;
}
