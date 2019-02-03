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
	void updateVariables(std::vector<Tensor>& variables, std::vector<Tensor>& gradients) override {
		int nvars = static_cast<int>(variables.size());
		CHECK_EQ(nvars, gradients.size());
//        if(newDeltaWeights_.size() != gradients.size()) {
//            for(int i = 0; i < gradients.size(); i++) {
//                newDeltaWeights_.push_back(gradients[i].scalarMult<T>(1.0f - momentum_));
//            }
//        } else {
//            for(int i = 0; i < gradients.size(); i++) {
//                DCHECK(newDeltaWeights_[i].hasSameShape(gradients[i]));
//                newDeltaWeights_[i].template asVec<T>().array() = gradients[i].asVec<T>().array() * static_cast<T>(1.0f - momentum_);
//            }
//        }
        newDeltaWeights2_.clear();
        for(int i = 0; i < nvars; i++) {
            if(!variables[i].hasSameShape(gradients[i])) {
//                std::vector<int> reduceIndices;
//                bool bcastable = BCast::reduceDims(reduceIndices,variables[i].shape(),gradients[i].shape());
//                CHECK(bcastable && (reduceIndices.size() == 1)) << "Variable and gradient should have the same shape: " << variables[i].dimString() << " vs " << gradients[i].dimString();
//                Eigen::array<int,1> reduceIndex = { reduceIndices[0]+1 };
//                typename TTypes<T,Tensor::MAX_DIMS>::Tensor gradient = gradients[i].tensorPadLeft<T,Tensor::MAX_DIMS>();
//                typename TTypes<T,Tensor::MAX_DIMS-1>::Tensor reduced = gradient.mean(reduceIndex);
//                newDeltaWeights2_.push_back(Tensor::tensorPadRight<T, Tensor::MAX_DIMS-1, Tensor::MAX_DIMS>(reduced) * static_cast<T>(1.0f - momentum_));
                CHECK(variables[i].hasSameShape(gradients[i])) << "Variable and gradient should have the same shape: " << variables[i].dimString() << " vs " << gradients[i].dimString();
            } else {
                newDeltaWeights2_.push_back(gradients[i].tensorPadRight<T, Tensor::MAX_DIMS>() * static_cast<T>(1.0f - momentum_));
            }
        }
//        for(int i = 0; i < nvars; i++) {
//            for(int j = 0; j < 3; j++) {
//                std::cout << newDeltaWeights2_[i].dimension(j);
//            }
//            std::cout << std::endl;
//        }
        
        // in the first run the oldDeltaWeights will be empty
        if(oldDeltaWeights2_.size() != nvars) {
            oldDeltaWeights2_.clear();
            for(int i = 0; i < nvars; i++) {
                oldDeltaWeights2_.push_back(newDeltaWeights2_[i]);
            }
        }
        
//        if(oldDeltaWeights_.size() != gradients.size()) {
//            for(int i = 0; i < newDeltaWeights2_.size(); i++) {
//                oldDeltaWeights_.push_back(newDeltaWeights_[i]);
//            }
//        }
//        for(int i = 0; i < newDeltaWeights_.size(); i++) {
//            newDeltaWeights_[i].template asVec<T>().array() = (oldDeltaWeights_[i].template asVec<T>().array() * static_cast<T>(momentum_)) + newDeltaWeights_[i].template asVec<T>().array();
//        }
        
        // dw <-- momentum * dw + (1 - momentum) * w
        for(int i = 0; i < nvars; i++) {
            newDeltaWeights2_[i] = (oldDeltaWeights2_[i] * static_cast<T>(momentum_)) + newDeltaWeights2_[i];
        }
        
        // update oldDeltaWeights
//        oldDeltaWeights_.clear();
//        for(int i = 0; i < nvars; i++) {
//            oldDeltaWeights_.push_back(newDeltaWeights_[i]);
//        }
        oldDeltaWeights2_.clear();
        for(int i = 0; i < nvars; i++) {
            oldDeltaWeights2_.push_back(newDeltaWeights2_[i]);
        }
        
//        for(int i = 0; i < nvars; i++) {
//            std::cout << variables[i].dimString() << std::endl;
//        }
//        std::cout << std::endl;
        
		for (int i = 0; i < nvars; i++) {
//    		CHECK(variables[i].hasSameShape(gradients[i])) << "Variable and gradient should have the same shape: " << variables[i].dimString() << " vs " << gradients[i].dimString();
//            if(!variables[i].hasSameShape(gradients[i]) {
//                std::vector<int> reduceIndices;
//                bool bcastable = BCast::reduceDims(reduceIndices,variables[i].shape(),gradients[i].shape());
//                CHECK(bcastable && (reduceIndices.size() == 1)) << "Variable and gradient should have the same shape: " << variables.[i].dimString() << " vs " << gradients[i].dimString();
//                
//            }
            variables[i].tensorPadRight<T, Tensor::MAX_DIMS>() = variables[i].tensorPadRight<T,Tensor::MAX_DIMS>() - (newDeltaWeights2_[i] * static_cast<T>(learningRate_));
		}
	}

private:
	float learningRate_;
    float momentum_;
    std::vector<Tensor> oldDeltaWeights_;
    std::vector<Tensor> newDeltaWeights_;
    std::vector<typename TTypes<T,Tensor::MAX_DIMS>::Tensor> newDeltaWeights2_;
    std::vector<typename TTypes<T,Tensor::MAX_DIMS>::Tensor> oldDeltaWeights2_;
};

inline NodePtr MomentumDescent(Graph& graph, NodePtr minimize, const float learningRate, const float momentum) {
    NodePtr ret;
    NUMBER_TYPE_CASES(minimize->dataType(), ret = std::make_shared<MomentumDescentOp<T>>(graph,minimize,learningRate,momentum));
    return ret;
}
