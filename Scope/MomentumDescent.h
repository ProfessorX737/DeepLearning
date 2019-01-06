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
	void updateVariables(std::vector<Tensor>& variables, const std::vector<Tensor>& gradients) override {
		int nvars = static_cast<int>(variables.size());
		CHECK_EQ(nvars, gradients.size());
        //std::vector<Tensor> newDeltaWeights;
        if(newDeltaWeights_.size() != gradients.size()) {
            for(int i = 0; i < gradients.size(); i++) {
                newDeltaWeights_.push_back(gradients[i].scalarMult<T>(1.0f - momentum_));
            }
        } else {
            for(int i = 0; i < gradients.size(); i++) {
                DCHECK(newDeltaWeights_[i].hasSameShape(gradients[i]));
                newDeltaWeights_[i].template asVec<T>().array() = gradients[i].asVec<T>().array() * static_cast<T>(1.0f - momentum_);
            }
        }
        if(oldDeltaWeights_.size() != gradients.size()) {
            for(int i = 0; i < newDeltaWeights_.size(); i++) {
                oldDeltaWeights_.push_back(newDeltaWeights_[i]);
            }
        }
        for(int i = 0; i < newDeltaWeights_.size(); i++) {
            newDeltaWeights_[i].template asVec<T>().array() = (oldDeltaWeights_[i].template asVec<T>().array() * static_cast<T>(momentum_)) + newDeltaWeights_[i].template asVec<T>().array();
        }
        
        // update oldDeltaWeights
        oldDeltaWeights_.clear();
        for(int i = 0; i < newDeltaWeights_.size(); i++) {
            oldDeltaWeights_.push_back(newDeltaWeights_[i]);
        }
		for (int i = 0; i < nvars; i++) {
    		CHECK(variables[i].shape().isSameShape(gradients[i].shape())) << "Variable and gradient should have the same shape: " << variables[i].dimString() << " vs " << gradients[i].dimString();
            variables[i].asVec<T>().array() = variables[i].asVec<T>().array() - (newDeltaWeights_[i].template asVec<T>().array() * static_cast<T>(learningRate_));
		}
	}

private:
	float learningRate_;
    float momentum_;
    std::vector<Tensor> oldDeltaWeights_;
    std::vector<Tensor> newDeltaWeights_;

};

inline NodePtr MomentumDescent(Graph& graph, NodePtr minimize, const float learningRate, const float momentum) {
    NodePtr ret;
    NUMBER_TYPE_CASES(minimize->dataType(), ret = std::make_shared<MomentumDescentOp<T>>(graph,minimize,learningRate,momentum));
    return ret;
}
