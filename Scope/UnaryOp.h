#pragma once
#include "Node.h"

class Tensor;

class UnaryOp : public Node {
public:
	UnaryOp(Node* operand, const std::string& class_name) : Node(class_name) {
		children_.push_back(operand);
	}
	Tensor eval() override {
		return unaryOp(children_[0]->eval());
	}
private:
	virtual Tensor unaryOp(const Tensor& operand) = 0;
};
