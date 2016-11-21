#include <iostream>
#include <iomanip>

using namespace std;

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
			sum = 0;
			for (int ii = i-1; ii <= i+1; ii++) {
				for (int jj = j-1; jj <= j+1; jj++) {
					sum += in[ii*height+jj] * filter[(ii-i+1)*3+jj-j+1];
				}
			}
			out[i*height+j] += sum;
			// cout << setw(3) << i*height+j << setw(5) << sum << endl;
		}
	}
	// cout << endl;
}

void convolution(float*** &data, float*** &filters, float** &output,
					int K, int C, int H, int W, int N) {
	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			// print_filter(filters[n][c]);
			// print_image(data[n][c], H, W);
			convolution_helper(data[n][c], filters[n][c], output[n], H, W);
		}
	}
}

int main(int argc, char const *argv[])
{
	int K, C, H, W, N;
	if (scanf("%d %d %d %d %d\n", &K, &C, &H, &W, &N) != 5) {
		cout << "ERROR: Invalid format\n";
		return 1;
	}

	float ***filters = new float**[K];
	// Read in data for filters
	for (int i = 0; i < K; i++) {
		filters[i] = new float*[C];
		for (int j = 0; j < C; j++) {
			filters[i][j] = new float[9];
			for (int m = 0; m < 3; m++) {
				for (int n = 0; n < 3; n++) {
					if (scanf("%f", &filters[i][j][m*3+n]) != 1) {
						cout << "ERROR: Invalid format\n";
						return 1;
					}
				// printf("%f ", filters[i][j][m*3+n]);
				}
				// cout << endl;
			}
			// cout << endl;
		}
	}

	// Read in data for image
	float ***data = new float**[N];
	for (int i = 0; i < N; i++) {
		data[i] = new float*[C];
		for (int j = 0; j < C; j++) {
			data[i][j] = new float[H*W];
			for (int m = 0; m < H; m++) {
				for (int n = 0; n < W; n++) {
				if (scanf("%f", &data[i][j][m*H+n]) != 1) {
					cout << "ERROR: Invalid format\n";
					return 1;
				}
				// printf("%f ", data[i][j][m*3+n]);
				}
				// cout << endl;
			}
			// cout << endl;
		}
	}

	// Create empty output object
	float **output = new float*[N];
	for (int i = 0; i < N; i++) {
		output[i] = new float[H*W]();
	}

	// Run the data
	convolution(data, filters, output, K, C, H, W, N);

	// Print the values stored in output
	for (int n = 0; n < N; n++) {
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				cout << setw(8) << output[n][i*H+j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}


	// Cleanup
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < C; j++) {
			delete [] filters[i][j];
		}
		delete [] filters[i];
	}
	delete [] filters;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < C; j++) {
			delete [] data[i][j];
		}
		delete [] data[i];
	}
	delete [] data;

	for (int i = 0; i < N; i++) {
		delete [] output[i];
	}
	delete [] output;
	return 0;
}
