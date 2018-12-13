#include "Variable.h"

VariablePtr Variable(Graph& graph, const Tensor::dim_init_list& shape, DataType dt) {
	return std::make_shared<VariableOp>(graph,shape, dt);
}

VariablePtr Variable(Graph& graph, const TensorShape& shape, DataType dt) {
	return std::make_shared<VariableOp>(graph,shape, dt);
}

VariableOp::VariableOp(Graph& graph, const Tensor::dim_init_list& shape, DataType dt) : Node(graph, "Variable") {
	t_.init(shape, dt);
}
VariableOp::VariableOp(Graph& graph, const TensorShape& shape, DataType dt) : Node(graph, "Variable") {
	t_.init(shape, dt);
}

Tensor VariableOp::tensor() {
	Tensor t;
	t = t_;
	return std::move(t);
}
