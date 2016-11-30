/* We are using 3 x 3 filters and an output tile size of 2 x 2. 
 * alpha = m + r - 1 = 4 */
#define m 2
#define r 3
#define alpha 4

/* For the filter g located at FILTERS[k][c], computes the transformation
 * u = G * g * G^T. Then, catters each matrix u into the output U. */
__kernel void filter_transform(__global float *filters,
        __constant float *G,
        __global float *U,
        int K,
        int C)
{

  size_t k = get_global_id(0);
  size_t c = get_global_id(1);

  if((int) k < K && (int) c < C) {
    int offset = (k * C + c) * r * r;

    /* Compute the matrix multiplication:
     * temp = G * filters[k][c] */
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

    /* Compute the matrix multiplication:
     * u = temp * G^T. Scatter u into U as follows:
     * U[xi][nu][k][c] = u[xi][nu]. */
    // TODO: rename vars appropriately
    offset = (k * C + c) * alpha * alpha;
    for(int xi = 0; xi < alpha; xi++){
      for(int nu = 0; nu < alpha; nu++) {
        sum = 0;
        for(int l = 0; l < r; l++) {
          sum += temp[xi*r + l] * G[nu*r + l];
        }
        U[xi*(alpha*K*C) + nu*(K*C) + k*C + c] = sum;
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
        int num_h_tiles,
        int num_w_tiles)
{
  int c = get_global_id(0);
  int block_y = get_global_id(1);
  int block_x = get_global_id(2);
  

  if (c < C && block_y < num_h_tiles && block_x < num_w_tiles) {
    int b = block_y * num_w_tiles + block_x;

    int x = block_x * m;
    int y = block_y * m;

    /* Compute the matrix multiplication:
     * temp = B^T * data[c][b], where b is a 1d index 
     * over the tiles in the image. */
    float temp[16]; // TODO: explain that we can hard code this ???
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

    /* Compute the matrix multiplication:
     * v = temp * B, then scatters v into V as follows:
     * V[xi][nu][c][b] = v[xi][nu] */
    // TODO: fix index variables
    int offset = c*(P*alpha*alpha) + b*(alpha*alpha);
    for(int xi = 0; xi < alpha; xi++) {
      for(int nu = 0; nu < alpha; nu++) {
        sum = 0;
        for(int l = 0; l < alpha; l++) {
          sum += temp[xi*alpha + l] * B[l*alpha + nu];
        }
        V[xi*(alpha*C*P) + nu*(C*P) + c*P + b] = sum;
      }
    }
  }
}

/* Computes U[xi][nu] * V[xi][ni], for each matrix in U and V,
 * where U has dimensions (alpha,alpha,K,C), and V has dimensions
 * (alpha,alpha,C,P). Stores U[xi][nu] * V[xi][ni] in M[xi][nu]*/
__kernel void calc_M (__global float *U,
        __global float *V,
        __global float *M,
        int K,
        int P,
        int C)
{
  int k = get_global_id(0);
  int b = get_global_id(1);
  if (k < K && b < P) {
    float sum;
    for(int xi = 0; xi < alpha; xi++) {
      for(int nu = 0; nu < alpha; nu++) {
        sum = 0;
        for(int c = 0; c < C; c++) {
          sum += U[xi*(alpha*K*C) + nu*(K*C) + k*C + c]
                   * V[xi*(alpha*C*P) + nu*(C*P) + c*P + b];
        }
        M[xi*(alpha*K*P) + nu*(K*P) + k*P + b] = sum;
      }
    }
  }
}

/* Gathers each matrix temp_m from M and computes A^T * temp_m * A.*/
__kernel void calc_Y(__global float *M,
        __constant float *A,
        __global float *Y,
        int out_H,
        int out_W,
        int K,
        int P,
        int num_h_tiles,
        int num_w_tiles)
{
  int k = get_global_id(0);
  int block_y = get_global_id(1);
  int block_x = get_global_id(2);
  
  if (k < K && block_y < num_h_tiles && block_x < num_w_tiles) {
    int b = block_y * num_w_tiles + block_x;
    float temp_m[16]; // alpha x alpha
    /* Gather temp_m from M, where:
     * temp_m[xi][nu] = M[xi][nu][k][b]*/
    for(int xi = 0; xi < alpha; xi++) {
      for(int nu = 0; nu < alpha; nu++) {
        temp_m[xi*alpha + nu] = M[xi*(alpha*K*P) + nu*(K*P)+ k*P + b]; //M[xi][nu][k][b]
      }
    }
    /* Compute temp_m = A^T * temp. */
    // A is alpha x 
    float temp[8]; // TODO explain size
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

    /* Compute Y[k][b] = temp_m * A. */
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
