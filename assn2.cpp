
#include "headers.h"

using namespace std;
using namespace cv;
// Using armadillo library
using namespace arma;

int main(int argc, char const *argv[]) {

    Facemash facemash;
    facemash.readData("train.txt");
    facemash.sub_mean();
    facemash.train_eigenfaces();
    facemash.save_model();


    ///////////////////////////// Model trained ////////////////////////////////

    /////// Testing  //////

    facemash.readData("test.txt");
    facemash.sub_mean();
    facemash.test_eigenfaces();

    facemash.print_accuracy();

    



    return 0;
}
