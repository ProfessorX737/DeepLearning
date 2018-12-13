#pragma once
#include "Node.h"
#include "logging.h"
#include "TensorShape.h"

class PlaceholderOp;

typedef std::shared_ptr<PlaceholderOp> PlaceholderPtr;
PlaceholderPtr Placeholder(Graph& graph, const TensorShape& shape, DataType dt = DT_FLOAT);
	

class PlaceholderOp : public Node, public std::enable_shared_from_this<PlaceholderOp> {
public:
	PlaceholderOp(Graph& graph, const TensorShape& shape, DataType dt = DT_FLOAT);
	void eval(Tensor& out) const override;
	void eval(std::unordered_map<int,Tensor>& nodeTensorMap, Tensor& out) const override;
	DataType dataType() const override { return dt_; }
	TensorShape shape() { return shape_; }
private:
	TensorShape shape_;
	DataType dt_;
};
