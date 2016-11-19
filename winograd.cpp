#include <iostream>
#include <fstream>
#include <armadillo>

using namespace std;
using namespace arma;

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

int main(int argc, char* argv[])
{
  if (argc != 2) {
    cout << "Usage: ./winograd <problem.in>";
  }
  ifstream file;
  file.open(argv[1]);
  float K, C, H, W, N;
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

  free(filters);
  free(images);
  return 0;
}
