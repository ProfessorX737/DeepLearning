#pragma once
#include "Node.h"
#include "logging.h"
#include "TensorShape.h"

class Placeholder : public Node {
public:
	Placeholder(Graph& graph) : Node("Placeholder",graph) {}
	Placeholder(Graph& graph, const TensorShape& shape) : Node("Placeholder",graph) {
		shape_ = std::move(shape);
	}
	void eval(Tensor& out) const override {
		LOG(FATAL) << "Placeholder with node id: " << getId() << ", needs to be fed a value";
	}
	void eval(std::unordered_map<int, Tensor>& nodeTensorMap, Tensor& out) const override {
		auto it = nodeTensorMap.find(getId());
		if (it == nodeTensorMap.end()) LOG(FATAL) << "Missing placeholder input";
		DCHECK(shape_.isSameShape(it->second.shape())) << "Input to placeholder is not have a compatible shape "
			<< shape_.dimString() << " vs " << it->second.shape().dimString();
		out = it->second;
	}
	TensorShape shape() { return shape_; }
private:
	TensorShape shape_;
};
