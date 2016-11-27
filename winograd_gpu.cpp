#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <sys/time.h>
#include "clhelp.h"

using namespace std;

int main(int argc, char *argv[])
{
  std::string kernel_source_str;
  
  /* Provide names of the OpenCL kernels
   * and cl file that they're kept in */
  std::string arraycompact_kernel_file = 
    std::string("winograd.cl");

  std::list<std::string> kernel_names;
  std::string filter_transform_name_str = std::string("filter_transform");

  kernel_names.push_back(filter_transform_name_str);

  cl_vars_t cv; 
  
  std::map<std::string, cl_kernel> 
    kernel_map;

  readFile(arraycompact_kernel_file,
     kernel_source_str);

  /* Intialize OpenCL runtime. */
  initialize_ocl(cv);

  /* Compile kernels. */
  compile_ocl_program(kernel_map, cv, 
          kernel_source_str.c_str(),
          kernel_names);

  int m = 2;
  int r = 3;
  int alpha = m + r - 1;

  if (argc != 3) {
    cout << "Usage: ./winograd_gpu <input filename> <output filename>\n";
  }
  ifstream file;
  file.open(argv[1]);
  int K, C, H, W;
  file >> K >> C >> H >> W;

  // Read in data for filters
  // float ***filters = new float**[K];
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

  // // Read in data for image
  // float **data = new float*[C];
  // for (int c = 0; c < C; c++) {
  //   data[c] = new float[H*W]();
  //   for (int i = 0; i < H; i++) {
  //     for (int j = 0; j < W; j++) {
  //       file >> data[c][i*H+j];
  //     }
  //   }
  // }
  file.close();

  float G[12] = {1.0, 0.0, 0.0,
                0.5, 0.5, 0.5,
                0.5, -0.5, 0.5,
                0.0, 0.0, 1.0};

  float *U = new float[K*C*alpha*alpha];

  // // Create empty output object
  // float **output = new float*[K];
  // for (int k = 0; k < K; k++) {
  //   output[k] = new float[H*W]();  
  // }

  cl_mem g_filters, g_data, g_U, g_G;

  /* Create buffers on GPU. */
  cl_int err = CL_SUCCESS;
  g_filters = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*K*C*3*3,NULL,&err);
  CHK_ERR(err);
  g_G = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*alpha*r,NULL,&err);
  CHK_ERR(err);
  g_U = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*K*C*alpha*alpha,NULL,&err);
  CHK_ERR(err);

  /* Copy data into buffers. */
  err = clEnqueueWriteBuffer(cv.commands, g_filters, true, 0,
           sizeof(float)*K*C*3*3, filters, 0, NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueWriteBuffer(cv.commands, g_G, true, 0,
           sizeof(float)*alpha*r, G, 0, NULL, NULL);
  CHK_ERR(err);

  cl_kernel filter_transform_kern = kernel_map[filter_transform_name_str];

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

  size_t global_work_size[2] = {K, C};
  size_t local_work_size[2] = {fmin(K, 64), C};

  err = clEnqueueNDRangeKernel(cv.commands,
         filter_transform_kern,
         2,//work_dim,
         NULL, //global_work_offset
         global_work_size, //global_work_size
         local_work_size, //local_work_size
         0, //num_events_in_wait_list
         NULL, //event_wait_list
         NULL //
         );
  CHK_ERR(err);

  err = clEnqueueReadBuffer(cv.commands, g_U, true, 0, sizeof(float)*K*C*alpha*alpha,
           U, 0, NULL, NULL);
  CHK_ERR(err);

  // for(int k = 0; k < K; k++) {
  //   for(int c = 0; c < C; c++) {
  //     printf("%d %d\n", k, c);
  //     for(int i = 0; i < alpha; i++) {
  //       for(int j = 0; j < alpha; j++) {
  //         int index = k*(C*alpha*alpha)+c*(alpha*alpha)+i*alpha+j;
  //         printf("%.8f ", U[index]);
  //       }
  //       printf("\n");
  //     }
  //   }
  // }

  clReleaseMemObject(g_filters); 
  clReleaseMemObject(g_U);

  uninitialize_ocl(cv);

  delete[] filters;
  // delete[] data;
  return 0;





}