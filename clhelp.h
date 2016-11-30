#ifndef __CLHELP_H
#define __CLHELP_H

#ifdef __linux__
#include "CL/cl.h"
#elif __APPLE__
#include <OpenCL/opencl.h>
#else
#error Unsupported OS
#endif

#include <map>
#include <cstdio>
#include <string>
#include <sstream>
#include <list>
#include <vector>
#include <cstdlib>
#include <cstring>

//#define DEBUG 1

typedef struct CLVARS
{
  cl_int err;
  cl_platform_id platform;
  cl_device_id device_id;
  cl_context context;
  cl_command_queue commands;

  cl_program main_program;
  std::list<cl_kernel> kernels;

  cl_uint platforms;

} cl_vars_t;

void ocl_device_query(cl_vars_t &cv);

std::string reportOCLError(cl_int err);

#define CHK_ERR(err) {\
  if(err != CL_SUCCESS) {\
    printf("Error: %s, File: %s, Line: %d\n", reportOCLError(err).c_str(), __FILE__, __LINE__); \
    exit(-1);\
  }\
}

void initialize_ocl(cl_vars_t& cv);
void uninitialize_ocl(cl_vars_t & clv);
void adjustWorkSize(size_t &global, size_t local);

void compile_ocl_program(cl_kernel & kernel, cl_vars_t &cv, const char * cl_src, 
			 const char * kname);

void compile_ocl_program(std::map<std::string, cl_kernel> &kernels, 
			 cl_vars_t &cv, const char * cl_src, 
			 std::list<std::string> knames);

void readFile(std::string& fileName, std::string &out); 
double timestamp();
#endif
