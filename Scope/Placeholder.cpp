#include "Placeholder.h"
#include "Tensor.h"

PlaceholderPtr Placeholder(Graph& graph, const TensorShape& shape, DataType dt) {
	return std::make_shared<PlaceholderOp>(graph, shape, dt);
}

PlaceholderOp::PlaceholderOp(Graph& graph, const TensorShape& shape, DataType dt) 
	: Node(graph, "Placeholder"), shape_(std::move(shape)), dt_(dt) {}

void PlaceholderOp::eval(std::unordered_map<int,Tensor>& nodeTensorMap, Tensor& out) const {
	auto it = nodeTensorMap.find(getId());
	if (it == nodeTensorMap.end()) LOG(FATAL) << "Missing placeholder input";
	DCHECK(shape_.isSameShape(it->second.shape())) << "Input to placeholder is not have a compatible shape "
		<< shape_.dimString() << " vs " << it->second.shape().dimString();
	out = it->second;
}
