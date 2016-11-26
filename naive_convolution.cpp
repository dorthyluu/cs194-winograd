#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/time.h>

using namespace std;

double timestamp();
void report_naive_statistics(int K, int C, int H, int W, double time);

void print_filter(float* filter) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      cout << setw(10) << filter[i*3+j];
    }
    cout << endl;
  }
  cout << endl;
}

void print_image(float* image, int H, int W) {
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      cout << setw(10) << image[i*H+j];
    }
    cout << endl;
  }
  cout << endl;
}

void convolution_helper(float* &in, float* &filter, float* &out, int height, int width) {
  float sum;
  for (int i = 1; i < height-1; i++) {
    for (int j = 1; j < width-1; j++) {
      for (int ii = i-1; ii <= i+1; ii++) {
        for (int jj = j-1; jj <= j+1; jj++) {
          out[i*height+j] += in[ii*height+jj] * filter[(ii-i+1)*3+jj-j+1];
        }
      }
    }
  }
}

void convolution(float** &data, float*** &filters, float** &output,
          int K, int C, int H, int W) {
  double time = timestamp();

  for (int c = 0; c < C; c++) {
    for (int k = 0; k < K; k++) {
      convolution_helper(data[c], filters[k][c], output[k], H, W);
    }
  }

  time = timestamp() - time;
  report_naive_statistics(K, C, H, W, time);
}

double timestamp()
{
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

void report_naive_statistics(int K, int C, int H, int W, double time) {
  int flop = (K * C * H * W * 3 * 3 * 2);
  double mflops = flop / (1024.0 * 1024.0 * time);
  cout << "Floating point operations: " << flop << "\n";
  cout << "Time Elapsed: " << time << "\n";
  cout << "MFlop/s: " << mflops << "\n";
}

int main(int argc, char const *argv[])
{
  if (argc != 3) {
    cout << "Usage: ./naive_convolution <input filename> <output filename>\n";
    return 1;
  }
  ifstream file;
  file.open(argv[1]);

  int K, C, H, W;
  file >> K >> C >> H >> W;

  // Read in data for filters
  float ***filters = new float**[K];
  for (int i = 0; i < K; i++) {
    filters[i] = new float*[C];
    for (int j = 0; j < C; j++) {
      filters[i][j] = new float[9];
      for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
          file >> filters[i][j][m*3+n];
        }
      }
    }
  }

  // Read in data for image
  float **data = new float*[C];
  for (int c = 0; c < C; c++) {
    data[c] = new float[H*W]();
    for (int m = 0; m < H; m++) {
      for (int n = 0; n < W; n++) {
        file >> data[c][m*H+n];
      }
    }
  }
  file.close();

  // Create empty output object
  float **output = new float*[K];
  for (int k = 0; k < K; k++) {
    output[k] = new float[H*W]();  
  }

  // Run the data
  convolution(data, filters, output, K, C, H, W);

  // Print the output to file
  ofstream fileout;
  fileout.open(argv[2], ofstream::out | ofstream::trunc );
  fileout << K << " " << C << " " << H << " " << W << " " << endl;
  for (int k = 0; k < K; k++) {
    for (int i = 1; i < H-1; i++) {
      for (int j = 1; j < W-1; j++) {
        fileout << setw(6) << setprecision(4) << output[k][i*H+j] << " ";
      }
      fileout << endl;
    }
    fileout << endl;
  }
  fileout.close();

  // Cleanup
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < C; j++) {
      delete [] filters[i][j];
    }
    delete [] filters[i];
  }
  delete [] filters;

  for (int c = 0; c < C; c++) {
    delete [] data[c];
  }
  delete [] data;

  for (int k = 0; k < K; k++) {
    delete [] output[k];
  }
  delete [] output;
  return 0;
}
