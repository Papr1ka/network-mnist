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
    net->loadDataset();
    
    //net->loadWeights();
    net->showLayers();
    net->calcAccuracy();
    net->training();
    net->calcAccuracy();
    net->saveWeights("Weights50k.txt");

    double* test = net->predict(net->dataset[1]);
    for (int i = 0; i < OUT_DIM; i++)
    {
        cout << test[i] << " ";
    }
    cout << argMax(test, 10) << endl;
    

    return 0;
}
