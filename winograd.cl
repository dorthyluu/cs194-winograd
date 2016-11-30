/* Computes G . filter . G^T for every filter in filters.
 * Write the output into U, which has not been scattered yet. */
__kernel void filter_transform(__global float *filters,
        __constant float *G,
        __global float *U,
        int K,
        int C,
        int r,
        int alpha)
{

  size_t k = get_global_id(0);
  size_t c = get_global_id(1);

  if((int) k < K && (int) c < C) {
    int offset = (k * C + c) * r * r;

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
        U[i*(alpha*K*C) + j*(K*C) + k*C + c] = sum;
      }
    }
  }
}

__kernel void data_transform(__global float *data,
        __constant float *B,
        __global float *V,
        int C,
        int P,
        int H,
        int W,
        int m,
        int alpha,
        int num_w_tiles)
{
  int c = get_global_id(0);
  int block_y = get_global_id(1);
  int block_x = get_global_id(2);
  int b = block_y * num_w_tiles + block_x;

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
        V[i*(alpha*C*P) + j*(C*P) + c*P + b] = sum;
      }
    }

  }
}

__kernel void calc_M (__global float *U,
        __global float *V,
        __global float *M,
        int K,
        int P,
        int C,
        int alpha)
{
  int k = get_global_id(0);
  int b = get_global_id(1);
  if (k < K && b < P) {
    float value;
    for(int xi = 0; xi < alpha; xi++) {
      for(int nu = 0; nu < alpha; nu++) {
        value = 0;
        for(int c = 0; c < C; c++) {
          value += U[xi*(alpha*K*C) + nu*(K*C) + k*C + c] * V[xi*(alpha*C*P) + nu*(C*P) + c*P + b];
        }
        M[xi*(alpha*K*P) + nu*(K*P) + k*P + b] = value;
      }
    }
  }
}

__kernel void calc_Y(__global float *M,
        __constant float *A,
        __global float *Y,
        int out_H,
        int out_W,
        int K,
        int P,
        int m,
        int alpha,
        int num_w_tiles)
{
  int k = get_global_id(0);
  int block_y = get_global_id(1);
  int block_x = get_global_id(2);

  int b = block_y * num_w_tiles + block_x;

  if (k < K && b < P) {
    float temp_m[16]; // alpha x alpha
    // gather
    for(int xi = 0; xi < alpha; xi++) {
      for(int nu = 0; nu < alpha; nu++) {
        temp_m[xi*alpha + nu] = M[xi*(alpha*K*P) + nu*(K*P)+ k*P + b]; //M[xi][nu][k][b]
      }
    }
    // A is alpha x m
    float temp[8];
    float sum;
    for(int i = 0; i < m; i++) {
      for(int j = 0; j < alpha; j ++) {
        sum = 0;
        for(int l = 0; l < alpha; l++) {
          sum += A[l*m + i] * temp_m[l*alpha + j];
        }
        temp[i*alpha + j] = sum;
      }
    }

    int x = block_x * m;
    int y = block_y * m;

    for(int i = 0; i < m; i++) {
      for(int j = 0; j < m; j ++) {
        sum = 0;
        for(int l = 0; l < alpha; l++) {
          sum += temp[i*alpha + l] * A[l*m + j];
        }
        Y[k*(out_H*out_W) + (y+i)*out_W + (x+j)] = sum;
      }
    }
  }
}
