#include <iostream>
#include "Tools.h"
#include "Network.h"

using namespace std;



int main(int argc, char* argv[])
{
    setlocale(LC_ALL, "Rus");

    cout << INPUT_DIM << endl;
    cout << HIDDEN_DIM << endl;
    cout << OUT_DIM << endl;
    
    Network* net = new Network();
    /*
    net->loadDataset();
    net->saveWeights("Weights1.txt");
    net->training();
    net->saveWeights("Weights2.txt");
    */
    double a[6]{2, 4, 6, 3, 5, 7};
    double b[2]{1, 8};
    double c[3]{1, 2, 3};
    double d[3]{ 1, 2, 3 };
    multiplyMatrix(3, 1, 2, a, b, c);
    for (int i = 0; i < 3; i++)
    {
        cout << c[i] << " ";
    }
    multiplyMatrix(2, 1, 3, a, d, c);
    for (int i = 0; i < 2; i++)
    {
        cout << c[i] << " ";
    }

    /*
    double* A = net->dataset[0];
    double* B = initMatrix(INPUT_DIM, HIDDEN_DIM);
    double* C = new double(HIDDEN_DIM);

    multiplyMatrix(1, HIDDEN_DIM, INPUT_DIM, A, B, C);

    for (int i = 0; i < HIDDEN_DIM; i++)
    {
        cout << C[i] << " ";
    }
    double* test = net->predict(net->dataset[1]);
    for (int i = 0; i < OUT_DIM; i++)
    {
        cout << test[i] << " ";
    }
    cout << argMax(test, 10) << endl;
    */

    return 0;
}
