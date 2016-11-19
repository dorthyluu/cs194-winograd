#include <iostream>
#include <fstream>
#include <armadillo>

using namespace std;
using namespace arma;


int main(int argc, char* argv[])
{
  if (argc != 2) {
    cout << "Usage: ./winograd <problem.in>";
  }
  ifstream file;
  file.open(argv[1]);
  float K, C, H, W, N;
  file >> K >> C >> H >> W >> N;
  mat** filters = (mat**) malloc(sizeof(mat*) * K);
  for (int i = 0; i < K; i++) {
    filters[i] = (mat*) malloc(sizeof(mat) * C);
    for (int j = 0; j < C; j++) {
      filters[i][j] = mat(3, 3);
      for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col ++) {
          file >> filters[i][j](row, col);
        }
      }
    }
  }

  mat** images = (mat**) malloc(sizeof(mat*) * N);
  for (int i = 0; i < N; i++) {
    images[i] = (mat*) malloc(sizeof(mat) * C);
    for (int j = 0; j < C; j++) {
      images[i][j] = mat(H, W);
      for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col ++) {
          file >> images[i][j](row, col);
        }
      }
    }
  }
  cout << images[0][0](0, 0);



  return 0;
}
