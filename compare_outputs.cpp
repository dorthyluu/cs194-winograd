#include <iostream>
#include <fstream>
#include <cmath>

#define EPSILON 0.01

using namespace std;

int main(int argc, char const *argv[])
{
	if (argc != 3) {
    cout << "Usage: ./compare_outputs <file1.out> <file2.out>\n";
    return 1;
  }
  ifstream output1, output2;
  output1.open(argv[1]);
  output2.open(argv[2]);
  
  int K1, C1, H1, W1,
      K2, C2, H2, W2;
  output1 >> K1 >> C1 >> H1 >> W1;
  output2 >> K2 >> C2 >> H2 >> W2;
  
  if (K1 != K2 || C1 != C2 || H1 != H2 || W1 != W2) {
   cout << "Output files were not created from the same dimensions." << endl;
   return 1;
  }

  float val1, val2;

  for (int k = 0; k < K1; k++) {
    for (int h = 0; h < H1 - 2; h++) {
       for (int w = 0; w < W1 - 2; w++) {
          output1 >> val1;
          output2 >> val2;
          if (abs(val1 - val2) > EPSILON) {
            printf("Values %f and %f at [%d][%d][%d] do not match up.\n",
              val1, val2, k, h, w);
            return 1;
          }
       }
    }
  }
  output1.close();
  output2.close();
  cout << "Values match up!" << endl;
  return 0;
}
