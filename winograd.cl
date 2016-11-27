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