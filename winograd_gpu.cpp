#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <sys/time.h>
#include "clhelp.h"

using namespace std;

/* We are using 3 x 3 filters and an output tile size of 2 x 2. */ // will need to define these in kernels too
// #define m 2
// #define r 3

bool power_of_two(int x) {
  return ((x & (x-1)) == 0);
}

void report_winograd_statistics(int K, int C, int P, double time) {
  int flop = (K * C * (4 * 3 * 5) * 2 +
              C * P * (4 * 4 * 7) * 2 + 
              16 * K * P * (2 * C - 1) + 
              K * P * (2 * 4 * 7) * 2);
  double mflops = flop / (1024.0 * 1024.0 * time);
  cout << "Floating point operations: " << flop << "\n";
  cout << "Time Elapsed: " << time << "\n";
  cout << "MFlop/s: " << mflops << "\n";
}

int main(int argc, char *argv[])
{
  /* Check that program arguments are properly specified. */
  if (argc != 3) {
    cout << "Usage: ./winograd_gpu <input filename> <output filename>\n";
    return 0;
  }

  ifstream file;
  file.open(argv[1]);

  /* Parse problem size. */
  int K, C, H, W;
  file >> K >> C >> H >> W;

  /* Check that sizes are appropriate. */
  bool valid = true;
  if (C % 2 != 0 && C % 3 != 0)
    valid = false;
  if((H-2) % 16 != 0)
    valid = false;
  if((W-2) % 16 != 0)
    valid = false;
  if (!valid) {
    cout << "Please make sure that:\n";
    cout << "C (# channels) is a multiple of 2 or 3\n";
    cout << "H (height of image) minus 2 is a power of 2\n";
    cout << "W (width of image) minus 2 is a power of 2\n";
    file.close();
    return 0;
  }
  // TODO: m, r, and alpha should be in header file
  int m = 2;
  int r = 3;
  int alpha = m + r - 1;
  int out_H = H - r + 1;
  int out_W = W - r + 1;
  int num_h_tiles = ceil(out_H/m);
  int num_w_tiles = ceil(out_W/m);
  int P = num_h_tiles * num_w_tiles;

  /* Read in filters. */
  float *filters = new float[K*C*r*r];
  for (int k = 0; k < K; k++) {
    for (int c = 0; c < C; c++) {
      for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++) {
          file >> filters[k*(C*r*r) + c*(r*r) + i*r + j];
        }
      }
    }
  }

  /* Read in image. */
  float *data = new float[C*H*W];
  for (int c = 0; c < C; c++) {
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
        file >> data[c*(H*W) + i*H + j];
      }
    }
  }
  file.close();

  /* Filter transform matrix. */
  float G[12] = {1.0, 0.0, 0.0,
                 0.5, 0.5, 0.5,
                 0.5, -0.5, 0.5,
                 0.0, 0.0, 1.0};

  /* Data transform matrix. */
  float B[16] = {1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, -1.0, 1.0,
                 -1.0, 1.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, -1.0};

  /* Inverse transform matrix (to transform the output after it is computed).*/
  float A[8] = {1.0, 0.0,
                1.0, 1.0,
                1.0, -1.0,
                0.0, -1.0};
  
  /* Array to hold the output. */
  float *Y = new float[K*out_H*out_W];


  /* OpenCL setup. */
  std::string kernel_source_str;
  
  /* Provide names of the OpenCL kernels
   * and cl file that they're kept in. */
  std::string arraycompact_kernel_file = 
    std::string("winograd.cl");

  std::list<std::string> kernel_names;
  std::string filter_transform_name_str = std::string("filter_transform");
  std::string data_transform_name_str = std::string("data_transform");
  std::string calc_M_name_str = std::string("calc_M");
  std::string calc_Y_name_str = std::string("calc_Y");

  kernel_names.push_back(filter_transform_name_str);
  kernel_names.push_back(data_transform_name_str);
  kernel_names.push_back(calc_M_name_str);
  kernel_names.push_back(calc_Y_name_str);
 
  std::map<std::string, cl_kernel> kernel_map;
  readFile(arraycompact_kernel_file, kernel_source_str);

  /* Intialize OpenCL runtime. */
  cl_vars_t cv;
  initialize_ocl(cv);

  /* Compile kernels. */
  compile_ocl_program(kernel_map, cv, 
          kernel_source_str.c_str(),
          kernel_names);


  /* Create buffers on GPU. */
  cl_mem g_filters, g_data, g_G, g_B, g_A, g_U, g_V, g_M, g_Y;

  cl_int err = CL_SUCCESS;
  g_filters = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*K*C*3*3,NULL,&err);
  CHK_ERR(err);
  g_data = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*C*H*W,NULL,&err);
  CHK_ERR(err);
  g_G = clCreateBuffer(cv.context,CL_MEM_READ_ONLY,
           sizeof(float)*alpha*r,NULL,&err);
  CHK_ERR(err);
  g_B = clCreateBuffer(cv.context,CL_MEM_READ_ONLY,
           sizeof(float)*alpha*alpha,NULL,&err);
  CHK_ERR(err);
  g_A = clCreateBuffer(cv.context,CL_MEM_READ_ONLY,
           sizeof(float)*alpha*m,NULL,&err);
  CHK_ERR(err);
  /* Will hold output of the filter transform. */
  g_U = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*K*C*alpha*alpha,NULL,&err);
  CHK_ERR(err);
  /* Will hold output of the data transform. */
  g_V = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*C*P*alpha*alpha,NULL,&err);
  CHK_ERR(err);
  /* Will hold the pre-transformed output. */
  g_M = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*K*P*alpha*alpha,NULL,&err);
  CHK_ERR(err);
  /* Will hold the final (transformed) output. */
  g_Y = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*K*out_H*out_W,NULL,&err);
  CHK_ERR(err);

  /* Copy data into buffers. */
  err = clEnqueueWriteBuffer(cv.commands, g_filters, true, 0,
           sizeof(float)*K*C*r*r, filters, 0, NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueWriteBuffer(cv.commands, g_data, true, 0,
           sizeof(float)*C*H*W, data, 0, NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueWriteBuffer(cv.commands, g_G, true, 0,
           sizeof(float)*alpha*r, G, 0, NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueWriteBuffer(cv.commands, g_B, true, 0,
           sizeof(float)*alpha*alpha, B, 0, NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueWriteBuffer(cv.commands, g_A, true, 0,
           sizeof(float)*alpha*m, A, 0, NULL, NULL);
  CHK_ERR(err);

  /* Compute global and local work sizes for the following: */

  /* Filter transform, which calculates U. */
  size_t global_work_size_U[2] = {(K+8)/8*8, C};
  size_t local_work_size_U[2] = {fmin(K, 8), C};

  /* Data transform, which calculates V. */
  size_t global_work_size_V[3] = {C, num_h_tiles, num_w_tiles};
  size_t local_work_size_V[3] = {C, fmin(num_w_tiles, 4), fmin(num_w_tiles, 4)};

  /* Calculating M. */
  int local_M = 8;
  size_t global_work_size_M[2] = {(K+local_M)/local_M*local_M, (P+local_M)/local_M*local_M};
  size_t local_work_size_M[2] = {local_M, local_M};

  /* Calculating Y. */
  size_t global_work_size_Y[3] = {(K+2)/2*2, num_h_tiles, num_w_tiles};
  size_t local_work_size_Y[3] = {fmin(K, 2), fmin(num_w_tiles, 8), fmin(num_w_tiles, 8)};

  /* Get the compiled kernels. */
  cl_kernel filter_transform_kern = kernel_map[filter_transform_name_str];
  cl_kernel data_transform_kern = kernel_map[data_transform_name_str];
  cl_kernel calc_M_kern = kernel_map[calc_M_name_str];
  cl_kernel calc_Y_kern = kernel_map[calc_Y_name_str];

  /* Set the arguments for each kernel. */
  err = clSetKernelArg(filter_transform_kern, 0, sizeof(cl_mem), &g_filters);
  CHK_ERR(err);
  err = clSetKernelArg(filter_transform_kern, 1, sizeof(cl_mem), &g_G);
  CHK_ERR(err);
  err = clSetKernelArg(filter_transform_kern, 2, sizeof(cl_mem), &g_U);
  CHK_ERR(err);
  err = clSetKernelArg(filter_transform_kern, 3, sizeof(int), &K);
  CHK_ERR(err);
  err = clSetKernelArg(filter_transform_kern, 4, sizeof(int), &C);
  CHK_ERR(err);
  err = clSetKernelArg(filter_transform_kern, 5, sizeof(int), &r);
  CHK_ERR(err);
  err = clSetKernelArg(filter_transform_kern, 6, sizeof(int), &alpha);
  CHK_ERR(err);

  err = clSetKernelArg(data_transform_kern, 0, sizeof(cl_mem), &g_data);
  CHK_ERR(err);
  err = clSetKernelArg(data_transform_kern, 1, sizeof(cl_mem), &g_B);
  CHK_ERR(err);
  err = clSetKernelArg(data_transform_kern, 2, sizeof(cl_mem), &g_V);
  CHK_ERR(err);
  err = clSetKernelArg(data_transform_kern, 3, sizeof(int), &C);
  CHK_ERR(err);
  err = clSetKernelArg(data_transform_kern, 4, sizeof(int), &P);
  CHK_ERR(err);
  err = clSetKernelArg(data_transform_kern, 5, sizeof(int), &H);
  CHK_ERR(err);
  err = clSetKernelArg(data_transform_kern, 6, sizeof(int), &W);
  CHK_ERR(err);
  err = clSetKernelArg(data_transform_kern, 7, sizeof(int), &m);
  CHK_ERR(err);
  err = clSetKernelArg(data_transform_kern, 8, sizeof(int), &alpha);
  CHK_ERR(err);
  err = clSetKernelArg(data_transform_kern, 9, sizeof(int), &num_w_tiles);
  CHK_ERR(err);

  err = clSetKernelArg(calc_M_kern, 0, sizeof(cl_mem), &g_U);
  CHK_ERR(err);
  err = clSetKernelArg(calc_M_kern, 1, sizeof(cl_mem), &g_V);
  CHK_ERR(err);
  err = clSetKernelArg(calc_M_kern, 2, sizeof(cl_mem), &g_M);
  CHK_ERR(err);
  err = clSetKernelArg(calc_M_kern, 3, sizeof(int), &K);
  CHK_ERR(err);
  err = clSetKernelArg(calc_M_kern, 4, sizeof(int), &P);
  CHK_ERR(err);
  err = clSetKernelArg(calc_M_kern, 5, sizeof(int), &C);
  CHK_ERR(err);
  err = clSetKernelArg(calc_M_kern, 6, sizeof(int), &alpha);
  CHK_ERR(err);

  err = clSetKernelArg(calc_Y_kern, 0, sizeof(cl_mem), &g_M);
  CHK_ERR(err);
  err = clSetKernelArg(calc_Y_kern, 1, sizeof(cl_mem), &g_A);
  CHK_ERR(err);
  err = clSetKernelArg(calc_Y_kern, 2, sizeof(cl_mem), &g_Y);
  CHK_ERR(err);
  err = clSetKernelArg(calc_Y_kern, 3, sizeof(int), &out_H);
  CHK_ERR(err);
  err = clSetKernelArg(calc_Y_kern, 4, sizeof(int), &out_W);
  CHK_ERR(err);
  err = clSetKernelArg(calc_Y_kern, 5, sizeof(int), &K);
  CHK_ERR(err);
  err = clSetKernelArg(calc_Y_kern, 6, sizeof(int), &P);
  CHK_ERR(err);
  err = clSetKernelArg(calc_Y_kern, 7, sizeof(int), &m);
  CHK_ERR(err);
  err = clSetKernelArg(calc_Y_kern, 8, sizeof(int), &alpha);
  CHK_ERR(err);
  err = clSetKernelArg(calc_Y_kern, 9, sizeof(int), &num_w_tiles);
  CHK_ERR(err);

  /* Start recording time for benchmarking. */
  double time = timestamp();

  /* Compute filter transform. */
  err = clEnqueueNDRangeKernel(cv.commands,
         filter_transform_kern,
         2,//work_dim,
         NULL, //global_work_offset
         global_work_size_U, //global_work_size
         local_work_size_U, //local_work_size
         0, //num_events_in_wait_list
         NULL, //event_wait_list
         NULL //
         );
  CHK_ERR(err);

  /* Compute data transform. */
  err = clEnqueueNDRangeKernel(cv.commands,
         data_transform_kern,
         3,//work_dim,
         NULL, //global_work_offset
         global_work_size_V, //global_work_size
         local_work_size_V, //local_work_size
         0, //num_events_in_wait_list
         NULL, //event_wait_list
         NULL //
         );
  CHK_ERR(err);

  /* Compute the pre-transformed output. */
  err = clEnqueueNDRangeKernel(cv.commands,
         calc_M_kern,
         2,//work_dim,
         NULL, //global_work_offset
         global_work_size_M, //global_work_size
         local_work_size_M, //local_work_size
         0, //num_events_in_wait_list
         NULL, //event_wait_list
         NULL //
         );
  CHK_ERR(err);

  /* Transform the output. */
  err = clEnqueueNDRangeKernel(cv.commands,
         calc_Y_kern,
         3,//work_dim,
         NULL, //global_work_offset
         global_work_size_Y, //global_work_size
         local_work_size_Y, //local_work_size
         0, //num_events_in_wait_list
         NULL, //event_wait_list
         NULL //
         );
  CHK_ERR(err);

  err = clFinish(cv.commands);
  CHK_ERR(err);

  time = timestamp() - time;

  /* Report timing and Mflop/s */
  report_winograd_statistics(K, C, P, time);

  err = clEnqueueReadBuffer(cv.commands, g_Y, true, 0, sizeof(float)*K*out_H*out_W,
           Y, 0, NULL, NULL);
  CHK_ERR(err);

  /* Write output Y to the specified file. */
  ofstream fileout;
  fileout.open(argv[2], ofstream::out | ofstream::trunc);
  fileout << K << " " << C << " " << H << " " << W << endl;
  for(int k = 0; k < K; k++) {
    fileout << "\n";
    for(int i = 0; i < out_H; i++) {
      for(int j = 0; j < out_W; j++) {
        int index = k*(out_H*out_W) + i*out_W + j;
        fileout << "   " << std::fixed << std::setw(5) << std::setprecision(4) << Y[index];
      }
      fileout << "\n";
    }
  }
  fileout.close();

  clReleaseMemObject(g_filters); 
  clReleaseMemObject(g_data);
  clReleaseMemObject(g_G);
  clReleaseMemObject(g_B);
  clReleaseMemObject(g_A);
  clReleaseMemObject(g_U);
  clReleaseMemObject(g_V);
  clReleaseMemObject(g_M);
  clReleaseMemObject(g_Y);

  uninitialize_ocl(cv);

  delete[] filters;
  delete[] data;
  delete[] Y;

  return 0;
}
