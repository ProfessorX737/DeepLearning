#pragma once
#include "Node.h"
#include "logging.h"
#include "Tensor.h"

template<int NINPUT>
class Op : public Node {
public:
	Op(const std::array<const Node*,NINPUT> operands, const std::string& class_name, Graph& graph) : Node(class_name, graph) {
		for (int i = 0; i < NINPUT; i++) {
			children_.push_back(std::move(operands[i]));
		}
	}
	void eval(Tensor& out) const override {
		std::array<Tensor, NINPUT> in;
		for (int i = 0; i < NINPUT; i++) {
			children_[i]->eval(in[i]);
		}
		op(std::move(in),out);
	}
	void eval(std::unordered_map<int, Tensor>& nodeTensorMap, Tensor& out) const override {

		std::array<Tensor, NINPUT> in;
		for (int i = 0; i < NINPUT; i++) {
			if (children_[i]->numChildren() == 0) {
				children_[i]->eval(nodeTensorMap, in[i]);
				nodeTensorMap[children_[i]->getId()] = in[i];
			}
			else {
				auto it = nodeTensorMap.find(children_[i]->getId());
				if (it == nodeTensorMap.end()) {
					children_[i]->eval(nodeTensorMap, in[i]);
					nodeTensorMap[children_[i]->getId()] = in[i];
				}
				else {
					in[i] = it->second;
				}
			}
		}
		op(std::move(in),out);
	}
private:
	virtual void op(const std::array<Tensor,NINPUT>& in, Tensor& out) const = 0;
};
