//
//  MomentumDescent.h
//  DeepLearning
//
//  Created by Xavier Poon on 4/01/2019.
//  Copyright © 2019 CreativityInk. All rights reserved.
//

#pragma once
#include "Optimizer.h"

template<typename T>
class MomentumDescentOp : public OptimizerOp<T> {
public:
	MomentumDescentOp(Graph& graph, NodePtr minimize, const float learningRate, const float momentum)
		: OptimizerOp<T>(graph,minimize), learningRate_(learningRate), momentum_(momentum) {
	}

	// requires: variables.size() == gradients.size()
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
        newDeltaWeights2_.clear();
        for(int i = 0; i < nvars; i++) {
            if(!variables[i].hasSameShape(gradients[i])) {
                CHECK(variables[i].hasSameShape(gradients[i])) << "Variable and gradient should have the same shape: " << variables[i].dimString() << " vs " << gradients[i].dimString();
            } else {
                newDeltaWeights2_.push_back(gradients[i].tensorPadRight<T, Tensor::MAX_DIMS>() * static_cast<T>(1.0f - momentum_));
            }
        }
        // in the first run the oldDeltaWeights will be empty
        if(oldDeltaWeights2_.size() != nvars) {
            oldDeltaWeights2_.clear();
            for(int i = 0; i < nvars; i++) {
                oldDeltaWeights2_.push_back(newDeltaWeights2_[i]);
            }
        }
        
        // dw <-- momentum * dw + (1 - momentum) * w
        for(int i = 0; i < nvars; i++) {
            newDeltaWeights2_[i] = (oldDeltaWeights2_[i] * static_cast<T>(momentum_)) + newDeltaWeights2_[i];
        }
        
        // update oldDeltaWeights
        oldDeltaWeights2_.clear();
        for(int i = 0; i < nvars; i++) {
            oldDeltaWeights2_.push_back(newDeltaWeights2_[i]);
        }
        
		for (int i = 0; i < nvars; i++) {
            variables[i].tensorPadRight<T, Tensor::MAX_DIMS>() = variables[i].tensorPadRight<T,Tensor::MAX_DIMS>() - (newDeltaWeights2_[i] * static_cast<T>(learningRate_));
		}
	}

private:
	float learningRate_;
    float momentum_;
    std::vector<typename TTypes<T,Tensor::MAX_DIMS>::Tensor> newDeltaWeights2_;
    std::vector<typename TTypes<T,Tensor::MAX_DIMS>::Tensor> oldDeltaWeights2_;
};

inline NodePtr MomentumDescent(Graph& graph, NodePtr minimize, const float learningRate, const float momentum) {
    NodePtr ret;
    NUMBER_TYPE_CASES(minimize->dataType(), ret = std::make_shared<MomentumDescentOp<T>>(graph,minimize,learningRate,momentum));
    return ret;
}
