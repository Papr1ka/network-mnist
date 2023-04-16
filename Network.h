#pragma once

#define DEBUG false

#if DEBUG

#define SIZE 3*3
#define INPUT_DIM SIZE
#define HIDDEN_DIM 4
#define OUT_DIM 2
#define ALPHA 0.01

#endif
#if not DEBUG

#define SIZE 28*28
#define INPUT_DIM SIZE
#define HIDDEN_DIM 128
#define OUT_DIM 10
#define ALPHA 0.001

#endif

#include "Tools.h"
#include <math.h>

#include <iostream>

using namespace std;

class Network
{
	double* weights1;
	double* bias1;
	double* weights2;
	double* bias2;

	double* relu(double* array, int n);
	double* softmax(double* array, int n);
	double drelu(double);
	double sparse_cross_entropy(double* array, int value);
	double* to_full(int value);

	/*
	double drelu(double);
	double* to_full(int value, int classes);
	*/
public:
	Network();
	~Network();

	void printWeights();
	double* predict(double* data);
	void loadDataset();
	double** dataset;

	int* dataAnswers;

	void training();
	double* losses;
	void calcAccuracy();
	void saveWeights(string path);
};