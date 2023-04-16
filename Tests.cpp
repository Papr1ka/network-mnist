#include "Tests.h"

int testRandom()
{
	int n = 3;
	int m = 5;
	double* array = new double[n * m];

	randomWeights(array, n * m);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if (array[i * n + j] < 0 or array[i * n + j] > 1)
			{
				return 1;
			}
			else
			{
				cout << array[i * n + j] << " ";
			}
		}
	}
	return 0;
}

int testAll()
{
	int value = 0;
	value = testRandom();
	if (value == 0)
	{
		cout << "Тест на рандомизацию пройден" << endl;
	}
	else
	{
		cout << "Тест на рандомизацию провален" << endl;
	}
	return value;
}

