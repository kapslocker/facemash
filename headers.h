#ifndef HEADERS_H
#define HEADERS_H
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
#include <limits.h>

using namespace std;
using namespace cv;
using namespace arma;

class Facemash{
    int n;                      // Dimension of the origin image.
    int N;                      // no. of training images.
    mat X;                      // Pattern matrix
    colvec mu;                  // mean vector
    mat Y,Y_test,W;             // coordinates and eigenvectors (TODO: for eigenfaces only as of now.)
    mat scatter;                // scatter matrix ( eigenfaces )
    vec closest_models;         // indices of training points closest to test points.
    vector<string> classes;     // store classes of train images
    vector<string> testClasses; // classes of test data points.
    int read_called;            // no. of times readData has been called.
 public:
     Facemash(){
         read_called = 0;
     }
    void readData(string f);
    void sub_mean();
    void train_eigenfaces();
    void save_model();

    void test_eigenfaces();
    void print_accuracy();
};


void Facemash::readData(string filename){
    read_called++;
    ifstream ti(filename,ios::in);
    string fname;
    cv::Mat m;
    vector<vector<double> >v;
    int temp=0;
    // This is to read the training data.
    while(getline(ti,fname)){
        if(read_called%2)
            classes.push_back(fname.substr(fname.find('\t')+1));
        else
            testClasses.push_back(fname.substr(fname.find('\t')+1));

        m = imread(fname.substr(0,fname.find('\t')),IMREAD_GRAYSCALE);
        m.convertTo(m,CV_64F);
        vector<double> array((double*)m.data, (double*)m.data + m.rows * m.cols);
        v.push_back(array);
    }
    ti.close();
    n = v[0].size();
    N = v.size();
    X.set_size(n,N);

    for(int i=0; i < v.size(); i++){
        X.col(i) = conv_to< colvec >::from(v[i]);
    }

}

void Facemash::sub_mean(){
    mu = mean(X,1);
    for(int i=0; i < X.n_cols; i++){
        X.col(i) -= mu;
    }
}

void Facemash::train_eigenfaces(){
        // Find eigenvectors of X'X and then use then for calculating eigenvectors of XX'
        // X has dimensions n x N . => X'X has dimensions NxN
        // and XX' has dimensions nxn (>> NxN).
        scatter = X.t()*X;
        cx_mat evec;
        cx_vec eval;
        eig_gen(eval,evec, scatter);
        mat eigvec_pseudo = conv_to< mat >::from(evec);
        vec eigval_pseudo = conv_to< vec >::from(eval);


        //Use these eigenvectors to get N eigenvalues of the covariance matrix.
        // U contains the weights to be multiplied to get points in a lower dimension.
        W = X * eigvec_pseudo; // U_i = X * V_i

        // normalise eigenvectors
        for(int i = 0;i < W.n_cols; i++){
            W.col(i)/=norm(W.col(i),"inf");
        }
        // We now have N eigenvectors which is less than n in all. These are the most important ones.


        //TODO: Select best K eigenvectors. on the basis of decreasing eigenvalues.
        int K = N;
        // get coordinates for the training data:
        Y = W.t() * X;
        cout<<"Weights calculated: " << W.n_rows << " x " << W.n_cols << endl;

}

void Facemash::save_model(){
    ofstream fout("train_model_eigenfaces.txt",ios::out);
    for(int i=0;i<W.n_rows;i++){
        for(int j = 0;j<W.n_cols;j++){
            fout<<W(i,j)<<" ";
        }
        fout<<"\n";
    }
    fout.close();
}

void Facemash::test_eigenfaces(){
    // weights for test data.
    Y_test = W.t() * X;
    //For each test vector, find the closest training vector.
    // Store the index of the closest training data point in the following matrix.
    closest_models.set_size(Y_test.n_cols);

    for(int i=0; i < Y_test.n_cols; i++){
        double min_dist = INT_MAX;
        for(int j = 0; j < Y.n_cols; j++){
            double dist = norm(Y_test.col(i) - Y.col(j));
            if(dist < min_dist){
                //TODO: Put threshold here if you wish.
                min_dist = dist;
                closest_models(i) = j;
            }

        }
    }
}
void Facemash::print_accuracy(){
    //Check accuracy now.
    int count = 0;
    for(int i = 0; i < Y_test.n_cols; i++){
        if(classes[closest_models[i]] == testClasses[i]){
            count++;
        }
    }
    cout << "The accuracy for eigenfaces is: " << ((double)count) / ((double)Y_test.n_cols) << endl;
}



#endif
