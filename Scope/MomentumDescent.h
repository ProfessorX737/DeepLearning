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
        std::vector<Tensor> newDeltaWeights;
        for(int i = 0; i < gradients.size(); i++) {
            newDeltaWeights.push_back(gradients[i].scalarMult<T>(1.0f - momentum_));
        }
        if(oldDeltaWeights_.size() == 0) {
            for(int i = 0; i < newDeltaWeights.size(); i++) {
                oldDeltaWeights_.push_back(newDeltaWeights[i]);
            }
        }
        for(int i = 0; i < newDeltaWeights.size(); i++) {
            newDeltaWeights[i].asVec<T>().array() = (oldDeltaWeights_[i].template asVec<T>().array() * static_cast<T>(momentum_)) + newDeltaWeights[i].asVec<T>().array();
        }
        
        // update oldDeltaWeights
        oldDeltaWeights_.clear();
        for(int i = 0; i < newDeltaWeights.size(); i++) {
            oldDeltaWeights_.push_back(newDeltaWeights[i]);
        }
		for (int i = 0; i < nvars; i++) {
    		CHECK(variables[i].shape().isSameShape(gradients[i].shape())) << "Adding two different shapes: "
    			<< variables[i].dimString() << " vs " << gradients[i].dimString();
            variables[i].asVec<T>().array() = variables[i].asVec<T>().array() - (newDeltaWeights[i].asVec<T>().array() * static_cast<T>(learningRate_));
		}
	}

private:
	float learningRate_;
    float momentum_;
    std::vector<Tensor> oldDeltaWeights_;

};

inline NodePtr MomentumDescent(Graph& graph, NodePtr minimize, const float learningRate, const float momentum) {
    NodePtr ret;
    NUMBER_TYPE_CASES(minimize->dataType(), ret = std::make_shared<MomentumDescentOp<T>>(graph,minimize,learningRate,momentum));
    return ret;
}
