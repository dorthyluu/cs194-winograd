/* Computes G . filter . G^T for every filter in filters.
 * Write the output into U, which has not been scattered yet. */
__kernel void filter_transform(__global float *filters,
        __global float *G,
        __global float *U,
        int K,
        int C,
        int r,
        int alpha)
{

  size_t k = get_global_id(0);
  size_t c = get_global_id(1);

  if((int) k < K && (int) c < C) {
    int offset = (k * C + c) * r * r; // increasing multiples of 9

    // temp = G * filters[k][c]
    float temp[12];
    float sum;
    for(int i = 0; i < alpha; i++) {
      for(int j = 0; j < r; j++) {
        sum = 0;
        for(int l = 0; l < r; l ++) {
          sum += G[i*r + l] * filters[offset + l*r + j];
        }
        temp[i*r + j] = sum;
      }
    }

    // U[k][c] = temp * G^T
    offset = (k * C + c) * alpha * alpha;
    for(int i = 0; i < alpha; i++){
      for(int j = 0; j < alpha; j++) {
        sum = 0;
        for(int l = 0; l < r; l++) {
          sum += temp[i*r + l] * G[j*r + l];
        }
        U[offset + i*alpha + j] = sum;
      }
    }
  }
}

/* Given an array that is of shape (d1, d2, d3, d4), scatters 
 * the array so that it has shape (d3, d4, d1, d2). */
__kernel void scatter(__global float *in,
        __global float *out,
        int d1,
        int d2,
        int d3,
        int d4)
{
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if ((int) i < d3 && (int) j < d4) {
    for(int k = 0; k < d1; k++) {
      for(int l = 0; l < d2; l++) {
        out[i*(d4*d1*d2) + j*(d1*d2) + k*d2 + l] = in[k*(d2*d3*d4) + l*(d3*d4)+ i*d4 +j];
      }
    }
  }
}

__kernel void data_transform(__global float *data,
        __global float *B,
        __global float *V,
        int C,
        int P,
        int H,
        int W,
        int m,
        int alpha)
{
  int c = get_global_id(0);
  int block_y = get_global_id(1);
  int block_x = get_global_id(2);
  int b = block_y * get_global_size(2) + block_x;

  if ((int) c < C && (int) b < P) {

    int x = block_x * m;
    int y = block_y * m;

    float temp[16];
    float sum;
    for(int i = 0; i < alpha; i++) {
      for(int j = 0; j < alpha; j++) {
        sum = 0;
        for(int l = 0; l < alpha; l++) {
          sum += B[l*alpha + i] * data[c*(H*W) + (y+l)*W + (x+j)];
        }
        temp[i*alpha + j] = sum;
      }
    }

    int offset = c*(P*alpha*alpha) + b*(alpha*alpha);
    for(int i = 0; i < alpha; i++) {
      for(int j = 0; j < alpha; j++) {
        sum = 0;
        for(int l = 0; l < alpha; l++) {
          sum += temp[i*alpha + l] * B[l*alpha + j];
        }
        V[offset + i*alpha + j] = sum;
      }
    }

  }
}
