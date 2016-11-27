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
  std::string scatter_name_str = std::string("scatter");
  std::string data_transform_name_str = std::string("data_transform");


  kernel_names.push_back(filter_transform_name_str);
  kernel_names.push_back(scatter_name_str);
  kernel_names.push_back(data_transform_name_str);

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

  if (argc != 3) {
    cout << "Usage: ./winograd_gpu <input filename> <output filename>\n";
  }
  ifstream file;
  file.open(argv[1]);
  int K, C, H, W;
  file >> K >> C >> H >> W;
  int m = 2;
  int r = 3;
  int alpha = m + r - 1;
  int out_H = H - r + 1;
  int out_W = W - r + 1;
  int num_h_tiles = ceil(out_H/m);
  int num_w_tiles = ceil(out_W/m);
  int P = num_h_tiles * num_w_tiles;

  // Read in data for filters
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

  // Read in data for image
  float *data = new float[C*H*W];
  for (int c = 0; c < C; c++) {
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
        file >> data[c*(H*W) + i*H + j];
        // printf("%d %d %d %.5f\n", c, i, j, data[c*(H*W) + i*H + j]);
      }
    }
  }
  file.close();

  float G[12] = {1.0, 0.0, 0.0,
                 0.5, 0.5, 0.5,
                 0.5, -0.5, 0.5,
                 0.0, 0.0, 1.0};

  float B[16] = {1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, -1.0, 1.0,
                 -1.0, 1.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, -1.0};

  float *U = new float[alpha*alpha*K*C];
  float *V = new float[alpha*alpha*C*P];

  // // Create empty output object
  // float **output = new float*[K];
  // for (int k = 0; k < K; k++) {
  //   output[k] = new float[H*W]();  
  // }

  cl_mem g_filters, g_data, g_G, g_B, g_U_temp, g_U, g_V_temp, g_V;

  /* Create buffers on GPU. */
  cl_int err = CL_SUCCESS;
  g_filters = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*K*C*3*3,NULL,&err);
  CHK_ERR(err);
  g_data = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*C*H*W,NULL,&err);
  CHK_ERR(err);
  g_G = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*alpha*r,NULL,&err);
  CHK_ERR(err);
  g_B = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*alpha*alpha,NULL,&err);
  CHK_ERR(err);
  g_U_temp = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*K*C*alpha*alpha,NULL,&err);
  CHK_ERR(err);
  g_U = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*K*C*alpha*alpha,NULL,&err);
  CHK_ERR(err);
  g_V_temp = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*C*P*alpha*alpha,NULL,&err); // ONLY NEED ONE TEMP!! THE BIGGER ONE
  CHK_ERR(err);
  g_V = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
           sizeof(float)*C*P*alpha*alpha,NULL,&err);
  CHK_ERR(err);

  /* Copy data into buffers. */
  err = clEnqueueWriteBuffer(cv.commands, g_filters, true, 0,
           sizeof(float)*K*C*3*3, filters, 0, NULL, NULL);
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

  /* Compute U. */
  cl_kernel filter_transform_kern = kernel_map[filter_transform_name_str];

  err = clSetKernelArg(filter_transform_kern, 0, sizeof(cl_mem), &g_filters);
  CHK_ERR(err);
  err = clSetKernelArg(filter_transform_kern, 1, sizeof(cl_mem), &g_G);
  CHK_ERR(err);
  err = clSetKernelArg(filter_transform_kern, 2, sizeof(cl_mem), &g_U_temp);
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

  // err = clEnqueueReadBuffer(cv.commands, g_U_temp, true, 0, sizeof(float)*K*C*alpha*alpha,
  //          U, 0, NULL, NULL);
  // CHK_ERR(err);

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

  /* Scatter U into its appropriate form. */
  cl_kernel scatter_kern = kernel_map[scatter_name_str];

  err = clSetKernelArg(scatter_kern, 0, sizeof(cl_mem), &g_U_temp);
  CHK_ERR(err);
  err = clSetKernelArg(scatter_kern, 1, sizeof(cl_mem), &g_U);
  CHK_ERR(err);
  err = clSetKernelArg(scatter_kern, 2, sizeof(int), &K);
  CHK_ERR(err);
  err = clSetKernelArg(scatter_kern, 3, sizeof(int), &C);
  CHK_ERR(err);
  err = clSetKernelArg(scatter_kern, 4, sizeof(int), &alpha);
  CHK_ERR(err);
  err = clSetKernelArg(scatter_kern, 5, sizeof(int), &alpha);
  CHK_ERR(err);

  global_work_size[0] = alpha;
  global_work_size[1] = alpha;
  local_work_size[0] = alpha;
  local_work_size[1] = alpha;

  err = clEnqueueNDRangeKernel(cv.commands,
         scatter_kern,
         2,//work_dim,
         NULL, //global_work_offset
         global_work_size, //global_work_size
         local_work_size, //local_work_size
         0, //num_events_in_wait_list
         NULL, //event_wait_list
         NULL //
         );
  CHK_ERR(err);

  // err = clEnqueueReadBuffer(cv.commands, g_U, true, 0, sizeof(float)*K*C*alpha*alpha,
  //          U, 0, NULL, NULL);
  // CHK_ERR(err);

  // for(int i = 0; i < alpha; i++) {
  //   for(int j = 0; j < alpha; j++) {
  //     printf("%d %d\n", i, j);
  //     for(int k = 0; k < K; k++) {
  //       for(int c = 0; c < C; c++) {
  //         int index = i*(alpha*K*C) + j*(K*C) + k*C + c;
  //         printf("%.8f ", U[index]);
  //       }
  //       printf("\n");
  //     }
  //   }
  // }

  /* Compute V. */
  cl_kernel data_transform_kern = kernel_map[data_transform_name_str];

  err = clSetKernelArg(data_transform_kern, 0, sizeof(cl_mem), &g_data);
  CHK_ERR(err);
  err = clSetKernelArg(data_transform_kern, 1, sizeof(cl_mem), &g_B);
  CHK_ERR(err);
  err = clSetKernelArg(data_transform_kern, 2, sizeof(cl_mem), &g_V_temp);
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

  size_t data_global_work_size[3] = {C, num_h_tiles, num_w_tiles};
  size_t data_local_work_size[3] = {C, num_h_tiles, num_w_tiles};

  err = clEnqueueNDRangeKernel(cv.commands,
         data_transform_kern,
         3,//work_dim,
         NULL, //global_work_offset
         data_global_work_size, //global_work_size
         data_local_work_size, //local_work_size
         0, //num_events_in_wait_list
         NULL, //event_wait_list
         NULL //
         );
  CHK_ERR(err);

  // err = clEnqueueReadBuffer(cv.commands, g_V_temp, true, 0, sizeof(float)*C*P*alpha*alpha,
  //          V, 0, NULL, NULL);
  // CHK_ERR(err);

  // for(int c = 0; c < C; c++) {
  //   for(int b = 0; b < P; b++) {
  //     printf("%d %d\n", c, b);
  //     for(int i = 0; i < alpha; i++) {
  //       for(int j = 0; j < alpha; j++) {
  //         int index = c*(P*alpha*alpha) + b*(alpha*alpha) + i*alpha + j;
  //         printf("%.8f ", V[index]);
  //       }
  //       printf("\n");
  //     }
  //   }
  // }

  err = clSetKernelArg(scatter_kern, 0, sizeof(cl_mem), &g_V_temp);
  CHK_ERR(err);
  err = clSetKernelArg(scatter_kern, 1, sizeof(cl_mem), &g_V);
  CHK_ERR(err);
  err = clSetKernelArg(scatter_kern, 2, sizeof(int), &C);
  CHK_ERR(err);
  err = clSetKernelArg(scatter_kern, 3, sizeof(int), &P);
  CHK_ERR(err);
  err = clSetKernelArg(scatter_kern, 4, sizeof(int), &alpha);
  CHK_ERR(err);
  err = clSetKernelArg(scatter_kern, 5, sizeof(int), &alpha);
  CHK_ERR(err);

  err = clEnqueueNDRangeKernel(cv.commands,
         scatter_kern,
         2,//work_dim,
         NULL, //global_work_offset
         global_work_size, //global_work_size
         local_work_size, //local_work_size
         0, //num_events_in_wait_list
         NULL, //event_wait_list
         NULL //
         );
  CHK_ERR(err);

  // err = clEnqueueReadBuffer(cv.commands, g_V, true, 0, sizeof(float)*C*P*alpha*alpha,
  //          V, 0, NULL, NULL);
  // CHK_ERR(err);

  // for(int i = 0; i < alpha; i++) {
  //   for(int j = 0; j < alpha; j++) {
  //     printf("%d %d\n", i, j);
  //     for(int c = 0; c < C; c++) {
  //       for(int b = 0; b < P; b++) {
  //         int index = i*(alpha*C*P) + j*(C*P) + c*P + b;
  //         printf("%.8f ", V[index]);
  //       }
  //       printf("\n");
  //     }
  //   }
  // }

  clReleaseMemObject(g_filters); 
  clReleaseMemObject(g_data);
  clReleaseMemObject(g_G);
  clReleaseMemObject(g_B);
  clReleaseMemObject(g_U_temp);
  clReleaseMemObject(g_U);
  clReleaseMemObject(g_V_temp);
  clReleaseMemObject(g_V);

  uninitialize_ocl(cv);

  delete[] filters;
  delete[] data;
  return 0;

  // status: write V transform kernel but data input format is wrong (need three channels for the image), cannot test yet



}