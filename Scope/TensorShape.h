#pragma once
#include <initializer_list>
#include <vector>
#include <Eigen/Dense>
#include "logging.h"
#include <string>

class TensorShape {
public:
	typedef Eigen::Index Dim;
	typedef std::initializer_list<Dim> dim_init_list;

	TensorShape() : num_elements_(1) {}
	TensorShape(const dim_init_list& dims);
	void init(const dim_init_list& dims);
	~TensorShape() {}

	void addDim(Dim d);
	int numDims() const { return dims_.size(); }
	int dimSize(int i) const;
	int numElements() const { return num_elements_; }
	std::string dimString() const;
	bool isSameShape(const TensorShape& other) const;
	bool operator==(const TensorShape& other) const { return isSameShape(other); }
	bool operator!=(const TensorShape& other) const { return !isSameShape(other); }

private:
	std::vector<Dim> dims_;
	int num_elements_;
};
