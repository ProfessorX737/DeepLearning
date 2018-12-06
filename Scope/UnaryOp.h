#pragma once
#include "Node.h"
#include "Tensor.h"


class UnaryOp : public Node {
public:
	UnaryOp(Node* operand, const std::string& class_name) : Node(class_name) {
		children_.push_back(operand);
	}
	void eval(Tensor& out) override {
		Tensor in;
		children_[0]->eval(in);
		unaryOp(in, out);
	}
private:
	virtual void unaryOp(Tensor& operand, Tensor& out) = 0;
};
