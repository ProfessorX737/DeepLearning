#pragma once
#include "Optimizer.h"
#include "Sub.h"

template<typename T>
class GradientDescentOp : public OptimizerOp<T> {
public:
	GradientDescentOp(Graph& graph, NodePtr minimize, float learningRate)
		: OptimizerOp(graph,minimize), learningRate_(learningRate) {
	}

	// requires: variables.size() == gradients.size()
	void updateVariables(std::vector<Tensor>& variables, const std::vector<Tensor>& gradients) const override {
		int nvars = static_cast<int>(variables.size());
		CHECK_EQ(nvars, gradients.size());
		for (int i = 0; i < nvars; i++) {
			SubOp<T>::subtract(variables[i], gradients[i].scalarMult<T>(learningRate_), variables[i]);
		}
	}
private:
	float learningRate_;
};
