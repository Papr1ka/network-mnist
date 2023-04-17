#include "Tools.h"

void randomWeights(double* array, int n)
{
    //Инициализирует матрицу рандомными значениями от 0 до 1
	for (int i = 0; i < n; i++)
	{
        array[i] = (double)(rand()) / RAND_MAX * 0.1;
	}
}

double* initMatrix(int n, int m)
{
    //Создаёт матрицу nxm и заполняет её числами от 0 до 1, возвращает указатель
    double* array = new double[n * m];

    randomWeights(array, n * m);
    return array;
}

void multiplyMatrix(int M, int N, int K, const double* A, const double* B, double* C)
{
    for (int i = 0; i < M; ++i)
    {
        double* c = C + i * N;
        for (int j = 0; j < N; ++j)
            c[j] = 0;
        for (int k = 0; k < K; ++k)
        {
            const double* b = B + k * N;
            double a = A[i * K + k];
            for (int j = 0; j < N; ++j)
                c[j] += a * b[j];
        }
    }
}

void multiplyMatrixNumber(double* Matrix, int size, double number)
{
    //Сумма матриц
    for (int i = 0; i < size; i++)
    {
        Matrix[i] = number * Matrix[i];
    }
}

void deleteMatrix(double* array)
{
    //удаляет матрицу
    if (array != nullptr)
    {
        delete[] array;
    }
}

void sumMatrix(int M, int N, const double* A, const double* B, double* C)
{
    //Сумма матриц
    int right = M * N;
    for (int i = 0; i < right ; i++)
    {
        C[i] = A[i] + B[i];
    }
}

void diffMatrix(int M, int N, const double* A, const double* B, double* C)
{
    //Разность матриц
    int right = M * N;
    for (int i = 0; i < right; i++)
    {
        C[i] = A[i] - B[i];
    }
}

#include <iostream>
#include <vector>

using namespace std;
int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

double** readMNIST(int NumberOfImages, int DataOfAnImage)
{
    double** data = new double*[NumberOfImages];

    for (int i = 0; i < NumberOfImages; i++)
    {
        data[i] = new double[DataOfAnImage];
    }

    ifstream file("t10k-images.idx3-ubyte", ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for (int i = 0; i < number_of_images; ++i)
        {
            for (int r = 0; r < n_rows; ++r)
            {
                for (int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    data[i][(n_rows * r) + c] = (double)temp / 255;
                }
            }
        }
        return data;
    }
    return nullptr;
}

int* readMNISTLabels(int NumberOfImages)
{
    int* data = new int [NumberOfImages];

    ifstream file("t10k-labels.idx1-ubyte", ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_items = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*)&number_of_items, sizeof(number_of_items));
        for (int i = 0; i < NumberOfImages; ++i)
        {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            data[i] = (int)temp;
        }
        return data;
    }
    return nullptr;
}

int argMax(double* value, int size)
{
    double max = value[0];
    int prediction = 0;
    double tmp;
    for (int j = 1; j < size; j++)
    {
        tmp = value[j];
        if (tmp > max) {
            prediction = j;
            max = tmp;
        }
    }
    return prediction;
}

double* transposeMatrix(int M, int N, double* Matrix)
{
    //Matrix MxN
    double* result = new double[M * N];

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            result[i * M + j] = Matrix[j * N + i];
        }
    }
    return result;
}

void statsArray(int n, double* array)
{
    double min = array[0];
    double max = array[0];
    double avg = 0;

    for (int i = 0; i < n; i++)
    {
        if (array[i] > max)
        {
            max = array[i];
        }
        if (array[i] < min)
        {
            min = array[i];
        }
        avg += array[i];
    }
    avg = avg / double(n);
    cout << "Массив [" << n << "], Min: " << min << " Max: " << max << " Avg: " << avg << endl;
}
