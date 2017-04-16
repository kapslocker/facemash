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
#include <map>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace arma;
map<string, int> cl_unique;     // for identifying the number of unique classes.
class Facemash{
    int n;                      // Dimension of the origin image.
    int N;                      // no. of training images.
    mat P;                      // Pattern matrix.
    mat X;                      // mean normalised pattern matrix.
    colvec mu;                  // mean vector.
    mat Y,Y_test,W;             // coordinates and eigenvectors (TODO: for eigenfaces only as of now.)
    mat scatter;                // scatter matrix ( eigenfaces )
    vec closest_models;         // indices of training points closest to test points.
    vector<string> classes;     // store classes of train images.
    vector<string> testClasses; // classes of test data points.
    int read_called;            // no. of times readData has been called.
    int C;                      // no. of classes.
    vec numClass;               // no. of training images in a class.
    mat cl_means;               // mean image for each class.
    vector< mat > pattern_class;// train images for each class. Used for S_w.
 public:
     Facemash(){
         read_called = 0;
     }
    void readData(string f);
    void sub_mean();
    mat eigenfaces(int);
    void fisherfaces();
    void save_model();

    void test();
    double accuracy();
    void class_means();
};


void Facemash::readData(string filename){
    read_called++;
    ifstream ti(filename,ios::in);
    string fname;
    cv::Mat m;
    vector<vector<double> >v;

    while(getline(ti,fname)){
        if(read_called%2){
            classes.push_back(fname.substr(fname.find('\t')+1));
        }
        else
            testClasses.push_back(fname.substr(fname.find('\t')+1));

        m = imread(fname.substr(0,fname.find('\t')),IMREAD_GRAYSCALE);
        m.convertTo(m,CV_64F);
        vector<double> array((double*)m.data, (double*)m.data + m.rows * m.cols);
        v.push_back(array);
    }
    n = v[0].size();
    N = v.size();
    P.set_size(n,N);
    ti.close();

    for(int i = 0; i < N; i++){
        if(v[i].size()!=v[0].size()){
            cout << i << " " << classes[i] << endl;
        }
    }
    //Fill in the pattern matrix using read data
    for(int i=0; i < v.size(); i++){
        P.col(i) = conv_to< colvec >::from(v[i]);
        cl_unique[classes[i]]++;                    // Use the hashmap to get number of unique classes and how many times they occur
    }
    C = cl_unique.size();
    X = P;  // do not modify the original pattern images.
}

void Facemash::sub_mean(){
    mu = mean(X,1);
    for(int i = 0; i < X.n_cols; i++){
        X.col(i) -= mu;
    }
}

// run PCA and return eigenvectors corresponding to first K eigenvalues.
mat Facemash::eigenfaces(int K = 30){
        // Find eigenvectors of X'X and then use then for calculating eigenvectors of XX'
        // X has dimensions n x N . => X'X has dimensions NxN
        // and XX' has dimensions nxn (>> NxN).
        scatter = X.t()*X;
        cx_mat evec;
        cx_vec eval;
        eig_gen(eval,evec, scatter);
        mat eigvec_pseudo = conv_to< mat >::from(evec);
        vec eigval_pseudo = conv_to< vec >::from(eval);

        // Select best K eigenvectors. on the basis of decreasing eigenvalues.
        mat eigvec(N,K);

        for(int i=0; i < K; i++){
            double max_val = -1;
            int index = -1;
            for(int j=0; j < eigval_pseudo.n_rows ; j++){
                if(eigval_pseudo[j] > max_val){
                    max_val = eigval_pseudo[j];
                    index = j;
                }
            }
            eigval_pseudo[index] = -1;
            eigvec.col(i) = eigvec_pseudo.col(index);
        }

        //Use these eigenvectors to get K eigenvalues of the covariance matrix.
        // We now have K eigenvectors which is less than n in all.
        // These are the most important ones.
        // W contains the weights to be multiplied to get points in a lower dimension.
        W = X * eigvec;
        // normalise eigenvectors
        for(int i = 0;i < W.n_cols; i++){
            W.col(i) /= norm(W.col(i),"inf");
        }

        // Traansformed feature space:
        Y = W.t() * X;
        cout << "Images transformed: " << Y.n_rows << " x " << Y.n_cols << endl;

        return W;
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

int q = 0;
void Facemash::test(){
    // weights for test data.
    if(q == 0){
        Y_test = W.t() * X;             // for eigenfaces
    }
    else
        Y_test = W.t() * P;             // for fisherfaces
    q++;
    //For each test vector, find the closest training vector.
    // Store the index of the closest training data point in the following matrix.
    closest_models.set_size(Y_test.n_cols);

    for(int i=0; i < Y_test.n_cols; i++){
        double min_dist = INT_MAX;
        for(int j = 0; j < Y.n_cols; j++){
            double dist = norm(Y_test.col(i) - Y.col(j));
            if(dist < min_dist){
                min_dist = dist;
                closest_models(i) = j;
            }

        }
    }
}

double Facemash::accuracy(){
    //Check accuracy now.
    int count = 0;
    for(int i = 0; i < Y_test.n_cols; i++){
        if(classes[closest_models[i]] == testClasses[i]){
            count++;
        }
    }
    return (((double)count) / ((double)Y_test.n_cols)) * 100;
}

void Facemash::class_means(){
    cl_means = zeros< mat >(n,C);
    numClass.set_size(C);
    for(int i = 0; i < C; i++){
        numClass[i] = 0;
    }
    for(int i = 0; i < P.n_cols; i++){
        int c = atoi(classes[i].c_str()) - 1;
        cl_means.col(c) += P.col(i);                // add in the values of corresponding class ke images
        numClass[c]++;                              // store the number of times we get an image from same class
    }

    for(int i = 0; i < C; i++){
        cl_means.col(i) /= (double)numClass[i];             // calculate per-class mean
    }
    mu = mean(P,1); // total mean.

    int indices[C];
    for(int i = 0; i < C; i++){
        indices[i] = 0;
        mat t = zeros< mat >(n,numClass[i]);
        pattern_class.push_back(t);
    }

    // classify images.
    for(int i = 0; i < N; i++){
        int cl = atoi(classes[i].c_str()) - 1;
        pattern_class[cl].col(indices[cl]++) = P.col(i);
    }

    // sub mean from each class.  i.e. for calculating (Xj - Uj)^2
    for(int i = 0; i < C; i++){
        for(int j = 0; j < numClass[i]; j++){
            pattern_class[i].col(j) -= cl_means.col(i);
        }
    }
}

void Facemash::fisherfaces(){
    mat W_pca = eigenfaces(N-C);

    // testing
    P = W_pca.t() * P;
    N = P.n_cols;
    n = P.n_rows;
    class_means();

    mat S_b = zeros< mat > (n,n);
    mat S_w = zeros< mat > (n,n);


//    #pragma omp parallel for num_threads(omp_get_max_threads()) shared(S_b , S_w)
    for(int i = 0; i < C; i++){
        S_w += pattern_class[i] * pattern_class[i].t();
        S_b = S_b + numClass[i] * (cl_means.col(i) - mu) * ((cl_means.col(i) - mu).t());
    }

    cout << "Scatter matrices evaluated"<<endl;

    cx_mat evec;
    cx_vec eval;
    eig_gen(eval, evec, inv(S_w) * S_b );
    mat eigvec_pseudo = conv_to< mat >::from(evec);
    vec eigval_pseudo = conv_to< vec >::from(eval);

    // Select best m = c - 1 eigenvectors. on the basis of decreasing eigenvalues.
    int m = C - 1;
    mat W_fld(N-C,m);

    for(int i=0; i < m; i++){
        double max_val = -1;
        int index = -1;
        for(int j=0; j < eigval_pseudo.n_rows ; j++){
            if(eigval_pseudo[j] > max_val){
                max_val = eigval_pseudo[j];
                index = j;
            }
        }
        eigval_pseudo[index] = -1;
        W_fld.col(i) = eigvec_pseudo.col(index);
    }

    W = W_pca * W_fld;          // n x m        // to be performed on test dataset.

    cout << "Saving model\n";
    save_model();
    cout << "Reducing dimensions\n";

    Y = W_fld.t() * P;              // m x N    // the train dataset has already been transformed by W_pca.
}



#endif
