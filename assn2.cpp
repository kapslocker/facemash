#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstddef>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <armadillo>

using namespace std;
using namespace cv;
// Using armadillo library
using namespace arma;

int main(int argc, char const *argv[]) {
    ifstream ti("train.txt",ios::in);
    string fname;
    cv::Mat m;
    int temp=0;
    vector<vector<double> > v;
    // This is to read the training data.
    while(getline(ti,fname)){
        m = imread(fname.substr(0,fname.find('\t')),IMREAD_GRAYSCALE);
        m.convertTo(m,CV_64F);
        vector<double> array((double*)m.data, (double*)m.data + m.rows * m.cols);
        v.push_back(array);
    }

    // X contains the images stacked column wise. Each column is a vector with image flattened out.
    arma::mat X(v[0].size(),v.size());

    for(int i=0;i<v.size();i++){
        colvec y = conv_to< colvec >::from(v[i]);
        X.col(i) = y;
    }

    int N;
    ti.close();
    return 0;
}
