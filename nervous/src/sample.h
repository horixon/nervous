//
//  sample.h
//  Neuralnet
//
//  data-tools ver. 01
//
//  Copyright (c) Microsoft Corporation
//
//  All rights reserved.
//
//  MIT License
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ""Software""), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#pragma once
#include <iterator>
#include <random>

namespace sample {

template<typename T, typename G>
class Bernoulli : public std::iterator<std::input_iterator_tag,T>
{
	typedef std::bernoulli_distribution D;

  const T *x;
  G& generator;

public: 
	Bernoulli(const T *x,G& generator):x(x),generator(generator) {}
  Bernoulli(const Bernoulli& bern) : x(bern.x),generator(bern.generator) {}

  Bernoulli& operator++() 
	{
    ++x;
  	return *this;
  }

  Bernoulli operator++(int) 
  {
  	Bernoulli<T,G> tmp(*this); 
  	operator++();
  	return tmp;
  }

  Bernoulli& operator=(const Bernoulli &rhs)
	{
		x = rhs.x;
    return *this;
	}
  	
  bool operator==(const Bernoulli& rhs) {return x==rhs.x;}
  bool operator!=(const Bernoulli& rhs) {return x!=rhs.x;}
  	
  const T operator*()
  {
  	D distribution(*x);
	 	return distribution(generator);
  }
};

template<typename T, typename G>
class Gaussian : public std::iterator<std::input_iterator_tag,T>
{
	typedef std::normal_distribution<T> D;

    const T *mu;
    const T *sigma;
    D distribution;
    G& generator;

public: 
	Gaussian(const T *mu, const T *sigma, G& generator): mu(mu), sigma(sigma), generator(generator) {}
  	Gaussian(const Gaussian& gauss): mu(gauss.mu), sigma(gauss.sigma), generator(gauss.generator) {}

  	Gaussian& operator++() 
  	{
  		++mu;
  		++sigma;
  		return *this;
  	}

  	Gaussian operator++(int) 
  	{
  		Gaussian<T,G> tmp(*this); 
  		operator++(); 
  		return tmp;
  	}

  	Gaussian& operator=(const Gaussian &rhs)
	{
		mu = rhs.mu;
		sigma = rhs.sigma;
        return *this;
	}
  	
  	bool operator==(const Gaussian& rhs) 
  	{
  		return mu == rhs.mu && sigma == rhs.sigma;
  	}
  	
  	bool operator!=(const Gaussian& rhs) 
  	{
		return mu != rhs.mu && sigma != rhs.sigma;
  	}
  	
  	const T operator*()
  	{
  		return (*mu) + (*sigma) * distribution(generator);
  	}
};

template<typename T, typename UC>
void sample(T *y, const unsigned count, UC unclamp)
{
	for(auto i=0; i<count;i++)
	{
		y[i] = *unclamp;
		unclamp++;
	}
}

template<typename T, typename G>
void saltandpepper(T p, std::vector<bool> &clamped, T *x, unsigned n, G& generator)
{
	T half = 0.5;
	Bernoulli<T,G> bentcoin(&p, generator);	
	Bernoulli<T,G> coin(&half, generator);

	for(int i = 0; i < n; i++)
	{
		if(*bentcoin){
			auto bw = *coin;
			clamped[i] = false;
			x[i] =  bw;
		}
	}	
}

}

