#pragma once

template<typename T>
class Initializer {
public:
	virtual void init(T* data, int num_elements) const = 0;
};

template<typename T>
class RandomNormal : public Initializer<T> {
public:
	RandomNormal(float mean, float stddev) : mean_(mean), stddev_(stddev) {}
	~RandomNormal() {}
	void init(T* data, int num_elements) const override {
		std::default_random_engine generator;
		std::normal_distribution<T> distribution(mean_, stddev_);
		for (int i = 0; i < num_elements; i++) {
			data[i] = distribution(generator);
		}
	}

private:
	float mean_;
	float stddev_;
};

template<typename T>
class ZeroInit : public Initializer<T> {
public:
	ZeroInit() {}
	~ZeroInit() {}
	void init(T* data, int num_elements) const override {
		memset(data, 0, num_elements * sizeof(T));
	}
};
