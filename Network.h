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
	double* weights1; //Матрица весов с входного слоя на скрытый
	double* bias1; //Сдвиги для скрытого слоя
	double* weights2; //Матрица весов с скрытого слоя на выходной
	double* bias2; //Сдвиги для выходного слоя
	int datasetSize = 60000; //Размер датасета, устанавливается при вызове loadWeights

	double* relu(double* array, int n); //Функция активации ReLu
	double* softmax(double* array, int n); //Функция активации Softmax
	double drelu(double); //Производная от ReLu
	double sparse_cross_entropy(double* array, int value); //Формула подсчёта ошибки, категориальная кросс энтропия
	double* to_full(int value); //Формирует массив вида [0, 0, ... , 0, 1, 0], размером OUT_DIM, где value - индекс числа 1

public:
	Network();
	~Network();

	void printWeights(); //Отладочная функция, выводит значения весов или сдвигом, менять в самой функции
	double* predict(double* data); //Функция предсказания, вход - матрица (одномерная) 28*28 пикселей - входное изображение цифры
	void loadDataset(DataSetType type); //Функция загрузки датасета MNIST
	double** dataset; //Датасет картинок

	int* dataAnswers; //Ответы к датасету

	void training(); //Функция обучения, обучает EPOCH_COUNT эпох по всему датасету
	double* losses; //Массив ошибок, формируется при training
	void calcAccuracy(); //Выводит процент ошибок с текущими весами в текущем датасете
	void saveWeights(); //Сохраняет веса в файл
	void loadWeights(); //Загружает веса из файла
	void showLayers(); //Показывает информацию о слоях
};