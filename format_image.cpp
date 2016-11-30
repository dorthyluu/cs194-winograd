#include <iostream>
#include <iomanip>
#include <fstream>
#include <CImg.h>

using namespace cimg_library;
using namespace std;

#define filterWidth 3
#define filterHeight 3

double emboss[filterHeight][filterWidth] =
{
   -2, -1, 0,
   -1, 1,  1,
   0,  1,  2
};

double edge_detect[filterHeight][filterWidth] =
{
   0,  1,  0,
   1, -4,  1,
   0,  1,  0
};

double identity[filterHeight][filterWidth] =
{
   0,  0,  0,
   0,  1,  0,
   0,  0,  0
};

double zero[filterHeight][filterWidth] =
{
   0,  0,  0,
   0,  0,  0,
   0,  0,  0
};

void print_3x3(ofstream &output, double (&matrix)[3][3]) {
  for (int h = 0; h < filterHeight; h++) {
    for (int w = 0; w < filterWidth; w++) {
      output << setw(3) << matrix[h][w];
    }
    output << endl;
  }
  output << endl;
}

void print_filter(ofstream &output, int C) {
	for (int i = 0; i < C; i++) {
		for (int c = 0; c < C; c++) {
			if (i == c) {
        print_3x3(output, emboss);
      } else {
        print_3x3(output, zero);
      }
    }
    output << endl;
	}
}


int main(int argc, char const *argv[]) {
	if (argc != 3) {
    cout << "Usage: ./format_image [image filename] [output filename]" << endl;
    return 1;
  }

  CImg<int> image(argv[1]);

  ofstream output;
  output.open(argv[2]);
  
  output << setw(4) << 3
  			 << setw(4) << image.spectrum()
  			 << setw(4) << image.height()
  			 << setw(4) << image.width() << endl;


  print_filter(output, image.spectrum());

	for (int n = 0; n < image.depth(); n++) {
		for (int c = 0; c < image.spectrum(); c++) {
			for (int h = 0; h < image.height(); h++) {
				for (int w = 0; w < image.width(); w++) {
					output << setw(4) << image(w, h, n, c);
				}
				output << endl;
			}
			output << endl;
		}
		output << endl;
	}
	output << endl;
  output.close();
  return 0;
}
