/*
 * mlmath.h
 *
 *  Created on: 29 giu 2016
 *      Author: vincenzo
 */

#ifndef MLMATH_H_
#define MLMATH_H_

#include "cmath"

using namespace std;

class MlMath{
public:
    static float sigmoid(float x){
    	//cout << x << endl;
        return  1/(1 + exp(-x));
    }

    static float sigmoid_derivative(float x){
        float sig = sigmoid(x);
        return sig*(1 - sig);
    }

    static float tanh(float x){
    	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }

    static float tanh_derivative(float x){
    	return (1 - (exp(x))*(exp(x)) );
    }

    static float ndim_sigmoid(int dim, float* x){
    	//x is the n-dimemsional array
    	//dim is the array dimension
    	//cout << "n_dim sigmoid: ";
    	//for(int i=0; i<dim; i++)
    		//cout << x[i] << " ";
    	//cout << endl;

    	float den = 1;
    	for(int i=0; i<dim; i++){

    		den = den*(1+exp(-x[i]));
    	}

    	return 1/den;
    }

    static float ndim_sigmoid_partial_der(int dim, float* x, int i){

    	float sig_der = sigmoid_derivative(x[i]);
    	float C = 1;

    	for(int j=0; j<dim; j++){
    		if(j!=i)
    			C = C * sigmoid(x[j]);
    	}
    	return C*sig_der;
    }

    static float relu(float x){
        if(x>0)
            return x;
        else
            return 0;
    }

    static float relu_derivative(float x){
        if(x>=0)
            return 1;
        else
            return 0;
    }
};

#endif /* MLMATH_H_ */
