#pragma once
#include "Optimizer.h"

template<typename T>
class GradientDescentOp : public OptimizerOp<T> {
public:
	GradientDescentOp(Graph& graph, NodePtr minimize, const float learningRate)
		: OptimizerOp<T>(graph,minimize), learningRate_(learningRate) {
	}
    void updateVariables(std::vector<Tensor>& variables, std::vector<Tensor>& gradients) override {
		int nvars = static_cast<int>(variables.size());
		CHECK_EQ(nvars, gradients.size());
		for (int i = 0; i < nvars; i++) {
    		CHECK(variables[i].shape().isSameShape(gradients[i].shape())) << "Subtracting two different shapes: "
    			<< variables[i].dimString() << " vs " << gradients[i].dimString();
            variables[i].asVec<T>().array() = variables[i].asVec<T>().array() - (gradients[i].asVec<T>().array() * static_cast<T>(learningRate_));
        }
    }
private:
	float learningRate_;
    float momentum_ = 0.5f;
    std::vector<Tensor> oldDeltaWeights_;

};

inline NodePtr GradientDescent(Graph& graph, NodePtr minimize, const float learningRate) {
    NodePtr ret;
    NUMBER_TYPE_CASES(minimize->dataType(), ret = std::make_shared<GradientDescentOp<T>>(graph,minimize,learningRate));
    return ret;
}
