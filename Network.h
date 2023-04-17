#pragma once

#define DEBUG false

#if DEBUG

#define SIZE 28*28
#define INPUT_DIM SIZE
#define HIDDEN_DIM 128
#define OUT_DIM 10
#define ALPHA 0.001
#define EPOCH_COUNT 1

#endif
#if not DEBUG

#define SIZE 28*28
#define INPUT_DIM SIZE
#define HIDDEN_DIM 128
#define OUT_DIM 10
#define ALPHA 0.001
#define EPOCH_COUNT 5

#endif

#include "Tools.h"
#include <math.h>

#include <iostream>

using namespace std;

class Network
{
	double* weights1; //������� ����� � �������� ���� �� �������
	double* bias1; //������ ��� �������� ����
	double* weights2; //������� ����� � �������� ���� �� ��������
	double* bias2; //������ ��� ��������� ����
	int datasetSize = 60000; //������ ��������, ��������������� ��� ������ loadWeights

	double* relu(double* array, int n); //������� ��������� ReLu
	double* softmax(double* array, int n); //������� ��������� Softmax
	double drelu(double); //����������� �� ReLu
	double sparse_cross_entropy(double* array, int value); //������� �������� ������, �������������� ����� ��������
	double* to_full(int value); //��������� ������ ���� [0, 0, ... , 0, 1, 0], �������� OUT_DIM, ��� value - ������ ����� 1

public:
	Network();
	~Network();

	void printWeights(); //���������� �������, ������� �������� ����� ��� �������, ������ � ����� �������
	double* predict(double* data); //������� ������������, ���� - ������� (����������) 28*28 �������� - ������� ����������� �����
	void loadDataset(DataSetType type); //������� �������� �������� MNIST
	double** dataset; //������� ��������

	int* dataAnswers; //������ � ��������

	void training(); //������� ��������, ������� EPOCH_COUNT ���� �� ����� ��������
	double* losses; //������ ������, ����������� ��� training
	void calcAccuracy(); //������� ������� ������ � �������� ������ � ������� ��������
	void saveWeights(); //��������� ���� � ����
	void loadWeights(); //��������� ���� �� �����
	void showLayers(); //���������� ���������� � �����
};