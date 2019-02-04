#pragma once
#include "Optimizer.h"

template<typename T>
class GradientDescentOp : public OptimizerOp<T> {
public:
	GradientDescentOp(Graph& graph, NodePtr minimize, const float learningRate)
		: OptimizerOp<T>(graph,minimize), learningRate_(learningRate) {
	}
    void updateVariables(std::vector<Tensor>& variables, std::vector<std::vector<Tensor>>& batchGradients) override {
        std::vector<Tensor> gradients;
        DCHECK_GT(batchGradients.size(),0);
		int nvars = static_cast<int>(variables.size());
        
        DCHECK_EQ(nvars,batchGradients[0].size());
        for(int i = 0; i < nvars; i++) {
            gradients.push_back(batchGradients[0][i]);
        }
        for(int i = 1; i < batchGradients.size(); i++) {
            CHECK_EQ(nvars, batchGradients[i].size());
            for(int j = 0; j < nvars; j++) {
                gradients[j].tensorPadRight<T, Tensor::MAX_DIMS>() = gradients[j].tensorPadRight<T,Tensor::MAX_DIMS>() + batchGradients[i][j].tensorPadRight<T, Tensor::MAX_DIMS>();
            }
        }
        for(int i = 0; i < nvars; i++) {
            gradients[i].tensorPadRight<T, Tensor::MAX_DIMS>() = gradients[i].tensorPadRight<T,Tensor::MAX_DIMS>() / static_cast<T>(batchGradients.size());
        }

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
