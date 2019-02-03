//
//  Broadcast.cpp
//  DeepLearning
//
//  Created by Xavier Poon on 12/01/2019.
//  Copyright © 2019 CreativityInk. All rights reserved.
//

#include "Broadcast.h"

bool BCast::compatible(const TensorShape& from, const TensorShape& to) {
    if(from.numDims() > to.numDims()) return false;
    DCHECK_LE(from.numDims(),Tensor::MAX_DIMS);
    DCHECK_LE(to.numDims(),Tensor::MAX_DIMS);
    CHECK_GE(from.numDims(),1);
    CHECK_GE(to.numDims(),1);
    int minNumDims = from.numDims();
    int diff = to.numDims() - from.numDims();
    for(int i = 0; i < minNumDims; i++) {
       if((from.dimSize(i) != to.dimSize(i+diff)) &&
          (from.dimSize(i) != 1)) {
           return false;
       }
    }
    return true;
}
// returns true if shape 'from' can be broadcasted to same shape as 'to'
bool BCast::multDimsPadLeft(Eigen::array<int,Tensor::MAX_DIMS>& multDims, const TensorShape& from, const TensorShape& to) {
    if(from.numDims() > to.numDims()) return false;
    DCHECK_LE(from.numDims(),Tensor::MAX_DIMS);
    DCHECK_LE(to.numDims(),Tensor::MAX_DIMS);
    CHECK_GE(from.numDims(),1);
    CHECK_GE(to.numDims(),1);
    int maxNumDims = to.numDims();
    int diff = to.numDims() - from.numDims();
    int start = Tensor::MAX_DIMS - maxNumDims;
    for(int i = 0; i < start; i++) {
        multDims[i] = 1;
    }
    for(int i = 0; i < maxNumDims; i++) {
        if((i < diff) || ((from.dimSize(i - diff) == 1) && (to.dimSize(i) != 1))) {
            multDims[i+start] = to.dimSize(i);
        } else if(from.dimSize(i - diff) == to.dimSize(i)){
            multDims[i+start] = 1;
        } else {
            return false;
        }
    }
    return true;
}

// returns true if shape 'from' can be broadcasted to same shape as 'to'
bool BCast::multDimsPadRight(Eigen::array<int,Tensor::MAX_DIMS>& multDims, const TensorShape& from, const TensorShape& to) {
    if(from.numDims() > to.numDims()) return false;
    DCHECK_LE(from.numDims(),Tensor::MAX_DIMS);
    DCHECK_LE(to.numDims(),Tensor::MAX_DIMS);
    CHECK_GE(from.numDims(),1);
    CHECK_GE(to.numDims(),1);
    int maxNumDims = to.numDims();
    int diff = to.numDims() - from.numDims();
    for(int i = 0; i < maxNumDims; i++) {
        if((i < diff) || ((from.dimSize(i - diff) == 1) && (to.dimSize(i) != 1))) {
            multDims[i] = to.dimSize(i);
        } else if(from.dimSize(i - diff) == to.dimSize(i)){
            multDims[i] = 1;
        } else {
            return false;
        }
    }
    for(int i = maxNumDims; i < Tensor::MAX_DIMS; i++) {
        multDims[i] = 1;
    }
    return true;
}

bool BCast::multDims(Eigen::array<int,Tensor::MAX_DIMS>& multDims, const TensorShape& from, const TensorShape& to) {
    return multDimsPadLeft(multDims,from,to);
}

bool BCast::reduceDims(std::vector<int>& reductionIndicies, const TensorShape& from, const TensorShape& to) {
    Eigen::array<int,Tensor::MAX_DIMS> multDims;
    if(!multDimsPadRight(multDims, from, to)) return false;
    for(int i = 0; i < multDims.size(); i++) {
        if(multDims[i] > 1) {
            reductionIndicies.push_back(i);
        }
    }
    return true;
}
