#include <iostream>
#include <fstream>
#include <armadillo>
#include <math.h>
#include <sys/time.h>

using namespace std;
using namespace arma;

double timestamp();
void report_winograd_statistics(int K, int C, int P, int N, double time);

mat** create_fourd_array(int d1, int d2, int d3, int d4) {
  mat** array = (mat**) malloc(sizeof(mat*) * d1);
  for (int i = 0; i < d1; i++) {
    array[i] = (mat*) malloc(sizeof(mat) * d2);
    for (int j = 0; j < d2; j++) {
      array[i][j] = mat(d3, d4);
    }
  }
  return array;
}

void free_fourd_array(mat** array, int d1) {
  for (int i = 0; i < d1; i++) {
    free(array[i]);
  }
  free(array);
}

mat** convolute(int K, int C, int H, int W, int N, mat** filters, mat** images) {
  int m = 2;
  int r = 3;
  int alpha = m + r - 1;
  int out_H = H - r + 1;
  int out_W = W - r + 1;
  int num_h_tiles = ceil(out_H/m);
  int num_w_tiles = ceil(out_W/m);
  int P = N * num_h_tiles * num_w_tiles;
  mat G = { {1.0, 0.0, 0.0},
            {0.5, 0.5, 0.5},
            {0.5, -0.5, 0.5},
            {0.0, 0.0, 1.0} };
  mat B = { {1, 0, 0, 0},
            {0, 1, -1, 1},
            {-1, 1, 1, 0},
            {0, 0, 0, -1} };
  mat A = { {1, 0},
            {1, 1},
            {1, -1},
            {0, -1}};

  auto gen_b = [num_h_tiles, num_w_tiles](int i, int y, int x) -> int {
    return i * num_h_tiles * num_w_tiles + y * num_w_tiles + x;
  };

  // factoring out malloc'ing before measuring runtime
  mat **U = create_fourd_array(alpha, alpha, K, C);
  mat **V = create_fourd_array(alpha, alpha, C, P);
  mat **M = (mat**) malloc(sizeof(mat*) * alpha);
  mat **Y = create_fourd_array(N, K, out_H, out_W);
  for (int xi = 0; xi < alpha; xi++) {
    M[xi] = (mat*) malloc(sizeof(mat) * alpha);
  }

  double time = timestamp();

  for (int k = 0; k < K; k++) {
    for (int c = 0; c < C; c++) {
      // flop: N * C * (4 * 3 * 5) * 2 
      mat u = G * filters[k][c] * G.t();
      for (int xi = 0; xi < alpha; xi++) {
        for (int nu = 0; nu < alpha; nu++) {
          U[xi][nu](k, c) = u(xi, nu);
        }
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int c = 0; c < C; c++) {
      for (int y = 0; y < num_h_tiles; y++) {
        for (int x = 0; x < num_w_tiles; x++) {
          mat d = images[i][c](span(y * m, y * m + alpha - 1), span(x * m, x * m + alpha - 1));
          // flop: C * P * (4 * 4 * 7) * 2
          mat v = B.t() * d * B;
          int b = gen_b(i, y, x);
          for (int xi = 0; xi < alpha; xi++) {
            for (int nu = 0; nu < alpha; nu++) {
              V[xi][nu](c, b) = v(xi, nu);
            }
          }
        }
      }
    }
  }

  
  for (int xi = 0; xi < alpha; xi++) {
    for (int nu = 0; nu < alpha; nu++) {
      // flop: 16 * K * P * (2C - 1) 
      M[xi][nu] = U[xi][nu] * V[xi][nu];
    }
  }

  mat m_hold = zeros<mat>(alpha, alpha);
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < K; k++) {
      for (int y = 0; y < num_h_tiles; y++) {
        for (int x = 0; x < num_w_tiles; x++) {
          int b = gen_b(i, y, x);
          for (int xi = 0; xi < alpha; xi++) {
            for (int nu = 0; nu < alpha; nu++) {
              m_hold(xi, nu) = M[xi][nu](k, b);
            }
          }
          // flop: N * K * P * (2 * 4 * 7) * 2
          Y[i][k](span(y*m, (y+1)*m-1), span(x*m, (x+1)*m-1)) = A.t() * m_hold * A;
        }
      }
    }
  }

  time = timestamp() - time;
  report_winograd_statistics(K, C, P, N, time);

  free_fourd_array(U, alpha);
  free_fourd_array(V, alpha);
  free_fourd_array(M, alpha);
  return Y;
}

double timestamp()
{
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

void report_winograd_statistics(int K, int C, int P, int N, double time) {
  int flop = (N * C * (4 * 3 * 5) * 2 +
                   C * P * (4 * 4 * 7) * 2 + 
                   16 * K * P * (2 * C - 1) + 
                   N * K * P * (2 * 4 * 7) * 2);
  double mflops = flop / (1024.0 * 1024.0 * time);
  cout << "Floating point operations: " << flop << "\n";
  cout << "Time Elapsed: " << time << "\n";
  cout << "MFlop/s: " << mflops << "\n";
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    cout << "Usage: ./winograd <input filename> <output filename>\n";
  }
  ifstream file;
  file.open(argv[1]);
  int K, C, H, W, N;
  file >> K >> C >> H >> W >> N;
  mat** filters = create_fourd_array(K, C, 3, 3);
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < C; j++) {
      for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col ++) {
          file >> filters[i][j](row, col);
        }
      }
    }
  }

  mat** images = create_fourd_array(N, C, H, W);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < C; j++) {
      for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col ++) {
          file >> images[i][j](row, col);
        }
      }
    }
  }
  file.close();

  mat** Y = convolute(K, C, H, W, N, filters, images);

  ofstream fileout;
  fileout.open(argv[2], ofstream::out | ofstream::trunc );
  fileout << K << " " << C << " " << H << " " << W << " " << N << endl;
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < K; k++) {
      fileout << Y[i][k] << "\n";
    }
    fileout << "\n";
  }
  fileout.close();

  free_fourd_array(Y, N);
  free_fourd_array(filters, K);
  free_fourd_array(images, N);
  return 0;
}
