#pragma once
#include <initializer_list>
#include <vector>
#include <Eigen/Dense>
#include <string>

class TensorShape {
public:
	typedef Eigen::Index Dim;
	typedef std::initializer_list<Dim> dim_init_list;

	TensorShape() : num_elements_(0) {}
	TensorShape(const dim_init_list& dims);
	void init(const dim_init_list& dims);
	~TensorShape() {}

	void addDim(Dim d);
	int numDims() const { return static_cast<int>(dims_.size()); }
	int dimSize(int i) const;
    int numElements() const;
	std::string dimString() const;
	bool isSameShape(const TensorShape& other) const;

private:
	std::vector<Dim> dims_;
	int num_elements_;
};
