#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sys/time.h>
#include <time.h>

#include "clhelp.h"

void initialize_ocl(cl_vars_t& cv)
{
  cv.err = clGetPlatformIDs(1, &(cv.platform), &(cv.platforms));
  CHK_ERR(cv.err);

  cv.err = clGetDeviceIDs(cv.platform, CL_DEVICE_TYPE_GPU, 1, &(cv.device_id), NULL);
  CHK_ERR(cv.err);

  cv.context = clCreateContext(0, 1, &(cv.device_id), NULL, NULL, &(cv.err));
  CHK_ERR(cv.err);

  cv.commands = clCreateCommandQueue(cv.context, cv.device_id, 
				     CL_QUEUE_PROFILING_ENABLE, &(cv.err));
  CHK_ERR(cv.err);


#ifdef DEBUG
  std::cout << "CL fill vars success" << std::endl;

  // Device info
  cl_ulong mem_size;
  cv.err = clGetDeviceInfo(cv.device_id, CL_DEVICE_GLOBAL_MEM_SIZE, 
			   sizeof(cl_ulong), &mem_size, NULL);

  std::cout << "Global mem size: " << mem_size << std::endl;

  size_t max_work_item[3];
  cv.err = clGetDeviceInfo(cv.device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, 
			   sizeof(max_work_item), max_work_item, NULL);

  std::cout << "Max work item sizes: " 
	    << max_work_item[0] << ", " 
	    << max_work_item[1] << ", " 
	    << max_work_item[2] 
	    << std::endl;
#endif
}

void uninitialize_ocl(cl_vars_t & clv)
{
  cl_int err;
  err = clFlush(clv.commands);
  CHK_ERR(err);

  for(std::list<cl_kernel>::iterator it = clv.kernels.begin();
      it != clv.kernels.end(); it++)
    {
      err = clReleaseKernel(*it);
      CHK_ERR(err);
    }
  clv.kernels.clear();
    
  err = clReleaseProgram(clv.main_program);
  CHK_ERR(err);
    
  err = clReleaseCommandQueue(clv.commands);
  CHK_ERR(err);

  err = clReleaseContext(clv.context);
  CHK_ERR(err);
}

void ocl_device_query(cl_vars_t &cv)
{
  char buf[256];
  cl_int err;
  memset(buf,0,sizeof(buf));
  err = clGetDeviceInfo(cv.device_id,CL_DEVICE_NAME,
			sizeof(buf),(void*)buf,
			NULL);
  CHK_ERR(err);
  printf("Running on a %s\n", buf);
}


void compile_ocl_program(cl_kernel & kernel, cl_vars_t &cv,
			 const char * cl_src, const char * kname)
{
  cl_int err;
  cv.main_program = clCreateProgramWithSource(cv.context, 1, 
					      (const char **) &cl_src, NULL, &err);
  //std::cout << cv.main_program << std::endl;
  CHK_ERR(err);

  err = clBuildProgram(cv.main_program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      size_t len;
      char buffer[2048];
      std::cout << "Error: Failed to build program executable: " << kname <<  std::endl;
      clGetProgramBuildInfo(cv.main_program, cv.device_id, CL_PROGRAM_BUILD_LOG, 
			    sizeof(buffer), buffer, &len);
      std::cout << buffer << std::endl;
      exit(1);
    }
  
  kernel = clCreateKernel(cv.main_program, kname, &(err));
  if(!kernel || err != CL_SUCCESS)
  {
    std::cout << "Failed to create kernel: " << kname  << std::endl;
    exit(1);
  }
  cv.kernels.push_back(kernel);
#ifdef DEBUG
  std::cout << "Successfully compiled " << kname << std::endl;
#endif
}

void compile_ocl_program(std::map<std::string, cl_kernel> &kernels, 
			 cl_vars_t &cv, const char * cl_src, 
			 std::list<std::string> knames)
{
  cl_int err;
  cv.main_program = clCreateProgramWithSource(cv.context, 1, (const char **) &cl_src, 
					      NULL, &err);
  CHK_ERR(err);

  err = clBuildProgram(cv.main_program, 0, NULL, NULL, NULL, NULL);

  if (err != CL_SUCCESS)
    {
      size_t len;
      char buffer[2048];
      std::cout << "Error: Failed to build program executable " << std::endl;
      clGetProgramBuildInfo(cv.main_program, cv.device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      std::cout << buffer << std::endl;
      exit(1);
    }
  
  for(std::list<std::string>::iterator it = knames.begin(); it != knames.end(); it++)
    {
      cl_kernel kernel = clCreateKernel(cv.main_program, (*it).c_str(), &(err));
      if(!kernel || err != CL_SUCCESS)
	{
	  std::cout << "Failed to create kernel: " << (*it).c_str()  << std::endl;
	  exit(1);
	}
#ifdef DEBUG
      std::cout << "Successfully compiled " << (*it).c_str() << std::endl;
#endif
      cv.kernels.push_back(kernel);
      kernels[*it] = kernel;
    }
}






void readFile(std::string& fileName, std::string &out)
{
  std::ifstream in(fileName.c_str(), std::ios::in | std::ios::binary);
  if(in)
    {
      in.seekg(0, std::ios::end);
      out.resize(in.tellg());
      in.seekg(0, std::ios::beg);
      in.read(&out[0], out.size());
      in.close();
    }
  else
    {
      std::cout << "Failed to open " << fileName << std::endl;
      exit(-1);
    }
}

double timestamp()
{
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

void adjustWorkSize(size_t &global, size_t local)
{
  if(global % local != 0)
    {
      global = ((global/local) + 1) * local;  
    }
}


std::string reportOCLError(cl_int err)
{
  std::stringstream stream;
  switch (err) 
    {
    case CL_DEVICE_NOT_FOUND:          
      stream << "Device not found.";
      break;
    case CL_DEVICE_NOT_AVAILABLE:           
      stream << "Device not available";
      break;
    case CL_COMPILER_NOT_AVAILABLE:     
      stream << "Compiler not available";
      break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:   
      stream << "Memory object allocation failure";
      break;
    case CL_OUT_OF_RESOURCES:       
      stream << "Out of resources";
      break;
    case CL_OUT_OF_HOST_MEMORY:     
      stream << "Out of host memory";
      break;
    case CL_PROFILING_INFO_NOT_AVAILABLE:  
      stream << "Profiling information not available";
      break;
    case CL_MEM_COPY_OVERLAP:        
      stream << "Memory copy overlap";
      break;
    case CL_IMAGE_FORMAT_MISMATCH:   
      stream << "Image format mismatch";
      break;
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         
      stream << "Image format not supported";    break;
    case CL_BUILD_PROGRAM_FAILURE:     
      stream << "Program build failure";    break;
    case CL_MAP_FAILURE:         
      stream << "Map failure";    break;
    case CL_INVALID_VALUE:
      stream << "Invalid value";    break;
    case CL_INVALID_DEVICE_TYPE:
      stream << "Invalid device type";    break;
    case CL_INVALID_PLATFORM:        
      stream << "Invalid platform";    break;
    case CL_INVALID_DEVICE:     
      stream << "Invalid device";    break;
    case CL_INVALID_CONTEXT:        
      stream << "Invalid context";    break;
    case CL_INVALID_QUEUE_PROPERTIES: 
      stream << "Invalid queue properties";    break;
    case CL_INVALID_COMMAND_QUEUE:          
      stream << "Invalid command queue";    break;
    case CL_INVALID_HOST_PTR:            
      stream << "Invalid host pointer";    break;
    case CL_INVALID_MEM_OBJECT:              
      stream << "Invalid memory object";    break;
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  
      stream << "Invalid image format descriptor";    break;
    case CL_INVALID_IMAGE_SIZE:           
      stream << "Invalid image size";    break;
    case CL_INVALID_SAMPLER:     
      stream << "Invalid sampler";    break;
    case CL_INVALID_BINARY:                    
      stream << "Invalid binary";    break;
    case CL_INVALID_BUILD_OPTIONS:           
      stream << "Invalid build options";    break;
    case CL_INVALID_PROGRAM:               
      stream << "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:  
      stream << "Invalid program executable";    break;
    case CL_INVALID_KERNEL_NAME:         
      stream << "Invalid kernel name";    break;
    case CL_INVALID_KERNEL_DEFINITION:      
      stream << "Invalid kernel definition";    break;
    case CL_INVALID_KERNEL:               
      stream << "Invalid kernel";    break;
    case CL_INVALID_ARG_INDEX:           
      stream << "Invalid argument index";    break;
    case CL_INVALID_ARG_VALUE:               
      stream << "Invalid argument value";    break;
    case CL_INVALID_ARG_SIZE:              
      stream << "Invalid argument size";    break;
    case CL_INVALID_KERNEL_ARGS:           
      stream << "Invalid kernel arguments";    break;
    case CL_INVALID_WORK_DIMENSION:       
      stream << "Invalid work dimension";    break;
      break;
    case CL_INVALID_WORK_GROUP_SIZE:          
      stream << "Invalid work group size";    break;
      break;
    case CL_INVALID_WORK_ITEM_SIZE:      
      stream << "Invalid work item size";    break;
      break;
    case CL_INVALID_GLOBAL_OFFSET: 
      stream << "Invalid global offset";    break;
      break;
    case CL_INVALID_EVENT_WAIT_LIST: 
      stream << "Invalid event wait list";    break;
      break;
    case CL_INVALID_EVENT:                
      stream << "Invalid event";    break;
      break;
    case CL_INVALID_OPERATION:       
      stream << "Invalid operation";    break;
      break;
    case CL_INVALID_GL_OBJECT:              
      stream << "Invalid OpenGL object";    break;
      break;
    case CL_INVALID_BUFFER_SIZE:          
      stream << "Invalid buffer size";    break;
      break;
    case CL_INVALID_MIP_LEVEL:             
      stream << "Invalid mip-map level";   
      break;  
    default: 
      stream << "Unknown";
      break;
    }
  return stream.str();
 }
