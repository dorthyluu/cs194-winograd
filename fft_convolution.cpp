#include <iostream>
#include <fstream>
#include <armadillo>
#include <math.h>
#include <sys/time.h>

using namespace std;
using namespace arma;

double timestamp();
void report_fft_statistics(int K, int C, int H, int W, double time);

mat** create_fourd_array(int d1, int d2, int d3, int d4) {
  mat** array = new mat*[d1]();
  for (int i = 0; i < d1; i++) {
    array[i] = new mat[d2]();
    for (int j = 0; j < d2; j++) {
      array[i][j] = mat(d3, d4);
    }
  }
  return array;
}

void free_fourd_array(mat** array, int d1) {
  for (int i = 0; i < d1; i++) {
    delete[] array[i];
  }
  delete[] array;
}

void convolute(int K, int C, int H, int W, cube* filters, cube& image, cube& result) {
  double time = timestamp();
  for (int i = 0; i < K; i++) {
    for (int c = 0; c < C; c++) {
      cx_mat fft_image = fft2(image.slice(c), H, W);
      cx_mat fft_filter = fft2(flipud(fliplr(filters[i].slice(c))), H, W);
      mat channel_out = real(ifft2(fft_image % fft_filter));
      result.slice(i) += channel_out(span(2, H - 1), span(2, W - 1));
    }
  }
  time = timestamp() - time;
  report_fft_statistics(K, C, H, W, time);
}

double timestamp()
{
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

void report_fft_statistics(int K, int C, int H, int W, double time) {
  long int flop = K * H * W * C + K * C * (H - 2) * (W - 2) +
    7 * (H * W) * log2(H * W) * K * C;
  double mflops = flop / (1024.0 * 1024.0 * time);
  cout << "Floating point operations: " << flop << "\n";
  cout << "Time Elapsed: " << time << "\n";
  cout << "MFlop/s: " << mflops << "\n";
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    cout << "Usage: ./fft_convolution <input filename> <output filename>\n";
  }
  ifstream file;
  file.open(argv[1]);
  int K, C, H, W;
  file >> K >> C >> H >> W;
  cube* filters = new cube[K]();
  for (int i = 0; i < K; i++) {
    filters[i] = cube(3, 3, C);
    for (int j = 0; j < C; j++) {
      for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col ++) {
          file >> filters[i](row, col, j);
        }
      }
    }
  }

  cube image = cube(H, W, C);
  for (int c = 0; c < C; c++) {
    for (int row = 0; row < H; row++) {
      for (int col = 0; col < W; col++) {
        file >> image(row, col, c);
      }
    }
  }

  file.close();

  cube result = cube(H-3+1, W-3+1, K, fill::zeros);
  convolute(K, C, H, W, filters, image, result);

  ofstream fileout;
  fileout.open(argv[2], ofstream::out | ofstream::trunc );
  fileout << K << " " << C << " " << H << " " << W << endl;
  for (int i = 0; i < K; i++) {
    fileout << result.slice(i) << "\n";
  }
  fileout.close();

  delete[] filters;
  return 0;
}
