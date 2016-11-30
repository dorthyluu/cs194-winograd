#include <iostream>
#include <iomanip>
#include <fstream>
#include <CImg.h>

using namespace cimg_library;
using namespace std;

int main(int argc, char const *argv[]) {
  if (argc != 3) {
    cout << "Usage: ./reform_image [input filename] [output filename]" << endl;
    return 1;
  }

  ifstream input;
  input.open(argv[1]);
  int K, C, H, W;
  input >> K >> C >> H >> W;


  CImg<float> image(H, W, 1, C);
  float val;
  cout << K << C << H << W <<endl;
	for (int c = 0; c < C; c++) {
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				input >> val;
				image(w, h, 0, c) = val;
			}
		}
	}

  input.close();
  image.normalize(0, 255);
  image.save(argv[2]);
  CImgDisplay main_disp(image, argv[1], 0);
  while (!main_disp.is_closed()) {
    main_disp.wait();
	}
  return 0;
}
