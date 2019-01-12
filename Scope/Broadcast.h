//
//  broadcast.h
//  DeepLearning
//
//  Created by Xavier Poon on 12/01/2019.
//  Copyright © 2019 CreativityInk. All rights reserved.
//

#ifndef BROADCAST_H
#define BROADCAST_H

#include "Tensor.h"

struct BCast {
    // returns true if shape 'from' can be broadcasted to same shape as 'to'
    static bool compatible(const TensorShape& from, const TensorShape& to);
    
    // returns true if shape 'from' can be broadcasted to same shape as 'to'
    static bool multDimsPadRight(Eigen::array<int,Tensor::MAX_DIMS>& multDims, const TensorShape& from, const TensorShape to);
    
    // returns true if shape 'from' can be broadcasted to same shape as 'to'
    static bool multDimsPadLeft(Eigen::array<int,Tensor::MAX_DIMS>& multDims, const TensorShape& from, const TensorShape to);
    
    // returns true if shape 'from' can be broadcasted to same shape as 'to'
    static bool multDims(Eigen::array<int,Tensor::MAX_DIMS>& multDims, const TensorShape& from, const TensorShape to);
};

#endif /* BROADCAST_H */
