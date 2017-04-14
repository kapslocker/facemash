
#include "headers.h"

using namespace std;
using namespace cv;
// Using armadillo library
using namespace arma;

int main(int argc, char const *argv[]) {


    Facemash facemash;
    facemash.readData("train.txt");
    ////////////////////////////// Eigenfaces //////////////////////////////////
    facemash.sub_mean();

    // for 30 best eigenvalues
    facemash.eigenfaces();
    // facemash.save_model();


    //////////////// Model trained ////////////////

    /////// Testing  //////

    facemash.readData("test.txt");
    facemash.sub_mean();
    facemash.test_eigenfaces();

    cout << "Eigenfaces: " << facemash.accuracy() << "%." << endl;


    ////////////////////////////// Fisherfaces /////////////////////////////////
    cout << "Creating fisherfaces model\n";
    facemash.readData("train.txt");
    facemash.class_means();
    facemash.fisherfaces();
    cout << " Model created and saved. Testing fisherfaces.\n";
    facemash.readData("test.txt");
    facemash.test_fisherfaces();
    cout << "Fisherfaces: " << facemash.accuracy() << "%.\n";

    return 0;
}
