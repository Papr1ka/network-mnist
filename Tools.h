#pragma once
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

void randomWeights(double* array, int n);

double* initMatrix(int n, int m);

void multiplyMatrix(int M, int N, int K, const double* A, const double* B, double* C);

void multiplyMatrixNumber(double* Matrix, int size, double number);

void deleteMatrix(double* array);

void sumMatrix(int M, int N, const double* A, const double* B, double* C);

void diffMatrix(int M, int N, const double* A, const double* B, double* C);

int ReverseInt(int i);

double** readMNIST(int NumberOfImages, int DataOfAnImage);

int* readMNISTLabels(int NumberOfImages);

int argMax(double* value, int size);

double* transposeMatrix(int M, int N, double* Matrix);

void statsArray(int n, double* array);
