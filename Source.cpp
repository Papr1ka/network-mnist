#include <iostream>
#include "Tools.h"
#include "Network.h"

using namespace std;

void trainOnWeights()
{
    //Обучение на ранее обученных весах, улучшение результата
    //инициализация
    Network* net = new Network();
    net->loadWeights(); //загрузка подсчитанных ранее весов

    //подсчёт исходной точности
    net->loadDataset(DataSetType::Test);
    net->calcAccuracy();

    //обучение
    net->loadDataset(DataSetType::Train);
    net->training();

    //подсчёт точности после тренировки
    net->loadDataset(DataSetType::Test);
    net->calcAccuracy();
    net->saveWeights();
}

void train()
{
    //Обучение с нуля
    //инициализация
    Network* net = new Network();

    //обучение
    net->loadDataset(DataSetType::Train);
    net->training();

    //подсчёт точности после тренировки
    net->loadDataset(DataSetType::Test);
    net->calcAccuracy();
    net->saveWeights();
}


int main(int argc, char* argv[])
{
    setlocale(LC_ALL, "Rus");

    cout << INPUT_DIM << endl;
    cout << HIDDEN_DIM << endl;
    cout << OUT_DIM << endl;
    
    Network* net = new Network();
    net->loadDataset(DataSetType::Test);
    net->loadWeights();
    double* array = readFile("image");
    statsArray(INPUT_DIM, array);
    if (array != nullptr)
    {
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                cout << array[i * 28 + j] << " ";
            }
            cout << endl;
        }
    }
    double* test = net->predict(array);
    for (int i = 0; i < OUT_DIM; i++)
    {
        cout << test[i] << " ";
    }
    cout << argMax(test, 10) << endl;
    
    /*
    net->calcAccuracy();
    net->showLayers();
    net->calcAccuracy();

    net->loadDataset(DataSetType::Train);
    net->training();

    net->loadDataset(DataSetType::Test);
    net->calcAccuracy();
    net->saveWeights();
    double* test = net->predict(net->dataset[1]);
    for (int i = 0; i < OUT_DIM; i++)
    {
        cout << test[i] << " ";
    }
    cout << argMax(test, 10) << endl;
    */

    return 0;
}
