/*******************************************************************************
Vendor: Xilinx
Associated Filename: vadd.cpp
Purpose: VITIS vector addition

*******************************************************************************
Copyright (C) 2019 XILINX, Inc.

This file contains confidential and proprietary information of Xilinx, Inc. and
is protected under U.S. and international copyright and other intellectual
property laws.

DISCLAIMER
This disclaimer is not a license and does not grant any rights to the materials
distributed herewith. Except as otherwise provided in a valid license issued to
you by Xilinx, and to the maximum extent permitted by applicable law:
(1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX
HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR
FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
in contract or tort, including negligence, or under any other theory of
liability) for any loss or damage of any kind or nature related to, arising under
or in connection with these materials, including for any direct, or any indirect,
special, incidental, or consequential loss or damage (including loss of data,
profits, goodwill, or any type of loss or damage suffered as a result of any
action brought by a third party) even if such damage or loss was reasonably
foreseeable or Xilinx had been advised of the possibility of the same.

CRITICAL APPLICATIONS
Xilinx products are not designed or intended to be fail-safe, or for use in any
application requiring fail-safe performance, such as life-support or safety
devices or systems, Class III medical devices, nuclear facilities, applications
related to the deployment of airbags, or any other applications that could lead
to death, personal injury, or severe property or environmental damage
(individually and collectively, "Critical Applications"). Customer assumes the
sole risk and liability of any use of Xilinx products in Critical Applications,
subject only to applicable laws and regulations governing limitations on product
liability.

THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT
ALL TIMES.

*******************************************************************************/
#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "vadd.h"
#include <stdio.h>
#include <math.h>


#include <stdint.h>
#include "time.h"
#include <sys/time.h>

static const int DATA_SIZE = 4096;
DATA_TYPE *values;
int       *col_indices;
int       *row_ptr;
DATA_TYPE *x;
DATA_TYPE *y_fpga;
DATA_TYPE *y_gold;
int row_size_val = 10;
int *row_size;
int col_size = 10;
int *nnz;
int nnz_val = 10;

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

#define M 1


double getTimestamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_usec + tv.tv_sec*1e6;
}


double software_start;
double software_end;
double software_execution_time;

double hardware_start;
double hardware_end;
double hardware_execution_time;


int read_mtx_SpMV(char* inFilename, int *row_size, int *col_size, int *nnz) {


	DATA_TYPE *cooValHostPtr = 0;
	int       *cooRowIndexHostPtr = 0;
	int       *cooColIndexHostPtr = 0;


	FILE *fp_input;

	int r;
	int c;
	DATA_TYPE v;


	fp_input = fopen(inFilename, "r");

	int nnzeo_false = 0;
	if (fp_input != NULL) {
		char line[1000];
		while (fgets(line, sizeof line, fp_input) != NULL) {// read a line from a file
			if (line[0] != '%') {
				sscanf(line, "%d %d %d", row_size, col_size, nnz);

				cooRowIndexHostPtr = (int *)malloc(*nnz * sizeof(int));
				cooColIndexHostPtr = (int *)malloc(*nnz * sizeof(int));
				cooValHostPtr = (DATA_TYPE *)malloc(*nnz * sizeof(DATA_TYPE));
				if ((!cooRowIndexHostPtr) || (!cooColIndexHostPtr) || (!cooValHostPtr)) {
					printf("Host malloc failed (matrix)\n");
					return 1;
				}


				int line_number = 0;
				while (fgets(line, sizeof line, fp_input) != NULL) {// read a line from a file

					sscanf(line, "%d %d %f", &r, &c, &v);
					if (v == 0) {
						//						printf("r = %d col = %d val=%lf\n", r, c, v);
						nnzeo_false++;
						continue;
					}

					r--;
					c--;
					//find row place
					(cooRowIndexHostPtr)[line_number] = r;
					(cooColIndexHostPtr)[line_number] = c;
					(cooValHostPtr)[line_number] = v;
					line_number++;
				}
			}
		}
	} else {
		perror(inFilename); //print the error message on stderr.
		exit(1);
	}

	*nnz = *nnz - nnzeo_false;

	int *cooRowIndexHostNewPtr = (int *)malloc(sizeof(int) * *nnz);
	int *cooColIndexHostNewPtr    = (int *)malloc(sizeof(int) * *nnz);
	DATA_TYPE *cooValHostNewPtr = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * *nnz);



	int index = 0;
	for (int i = 0; i < *row_size; i++) {
		for (int j = 0; j < *nnz; j++) {
			if (cooRowIndexHostPtr[j] == i) {
				cooRowIndexHostNewPtr[index] = cooRowIndexHostPtr[j];
				cooColIndexHostNewPtr[index] = cooColIndexHostPtr[j];
				cooValHostNewPtr[index] = cooValHostPtr[j];
				index++;
			}
		}
	}

//	for (int i = 0; i < *nnz; i++) {
//		printf("%d %d %f\n ", cooRowIndexHostNewPtr[i], cooColIndexHostNewPtr[i], cooValHostNewPtr[i]);
//	}


	int d = 0;
	int r_index = 0;
	for (r_index = 0; r_index < *row_size; r_index++) {
		int nonzero_line = 0;
		for (; d < *nnz; d++) {
			int current_row_index = cooRowIndexHostNewPtr[d];
			if (current_row_index == r_index) {
				row_ptr[r_index] = d;
				nonzero_line = 1;
				break;
			}
		}
		if (nonzero_line==0) {
			row_ptr[r_index] = row_ptr[r_index-1];
		}
	}
	row_ptr[r_index]=*nnz;
	for (int i = 0; i < *nnz; i++) {
		col_indices[i] = cooColIndexHostNewPtr[i];
		values[i]      = cooValHostNewPtr[i];
	}



//	for (int i = 0; i < *row_size; i++) {
//		printf("row_ptr[%d]=%d\n", i, row_ptr[i]);
//	}

	free(cooValHostPtr);
	free(cooRowIndexHostPtr);
	free(cooColIndexHostPtr);
	free(cooRowIndexHostNewPtr);

}

int generateSpMV(int row_size, int col_size, int *nnz) {

	int rand_index[II*M];

	int v_i = 0;
	int r_i = 0;
	int index_not_found = 1;



	for (int ix = 0; ix < row_size; ix++) {

		row_ptr[r_i] = v_i;
		int r_tmp = 0;
		for (int i = 0; i < II*M; i++) {
			index_not_found = 1;
			while(index_not_found) {
				int rand_col_index = rand()%col_size;
				index_not_found = 0;
				for (int s = 0; s < r_tmp; s++) {
					if (rand_index[s] == rand_col_index) {
						index_not_found = 1;
						break;
					}
				}
				if (index_not_found == 0)
					rand_index[r_tmp++] = rand_col_index;
			}
		}

		for (int i = 0; i < II*M; i++) {
			DATA_TYPE r = (10.0*(rand()+1)/RAND_MAX);
			values[v_i] = r;
			col_indices[v_i++] = rand_index[i];
		}
		r_i++;
	}

	*nnz = v_i--;
	row_ptr[r_i]=*nnz;
	return 0;
}



int gold_spmv() {

	int i=0, j=0, rowStart=0, rowEnd=row_size_val;
	long int k=0;
	DATA_TYPE y0=0.0;
	int last_j = 0;
	for (i = rowStart; i < rowEnd; ++i) {
		y0 = 0.0;
	    for (j = row_ptr[i] ; j < row_ptr[i+1]; ++j) {
	    	y0 += values[j] * x[col_indices[j]];
	    }

	    y_gold[i] = y0;
	 }


	return 0;
}

int main(int argc, char* argv[]) {

    //TARGET_DEVICE macro needs to be passed from gcc command line
    if(argc != 2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}
    std::string xclbinFilename = argv[1];
    
    // Compute the size of array in bytes
    size_t size_in_bytes = DATA_SIZE * sizeof(int);
    
    // Creates a vector of DATA_SIZE elements with an initial value of 10 and 32
    // using customized allocator for getting buffer alignment to 4k boundary
    
    std::vector<cl::Device> devices;
    cl::Device device;
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel krnl_sparse_mv;
    cl::Program program;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    //traversing all Platforms To find Xilinx Platform and targeted
    //Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if ( platformName == "Xilinx"){
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
	    if (devices.size()){
		    device = devices[0];
		    found_device = true;
		    break;
	    }
        }
    }
    if (found_device == false){
       std::cout << "Error: Unable to find Target Device " 
           << device.getInfo<CL_DEVICE_NAME>() << std::endl;
       return EXIT_FAILURE; 
    }

    // Creating Context and Command Queue for selected device
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    std::cout << "INFO: Reading " << xclbinFilename << std::endl;
    FILE* fp;
    if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
        printf("ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
        exit(EXIT_FAILURE);
    }
    // Load xclbin 
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    
    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    devices.resize(1);
    OCL_CHECK(err, program = cl::Program(context, devices, bins, NULL, &err));
    
    // This call will get the kernel object from program. A kernel is an 
    // OpenCL function that is executed on the FPGA. 
    OCL_CHECK(err, krnl_sparse_mv = cl::Kernel(program,"spmv_mohammad", &err));

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device. 
    OCL_CHECK(err, cl::Buffer buffer_x(context, CL_MEM_READ_ONLY, COL_SIZE_MAX*sizeof(float), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_values(context, CL_MEM_READ_ONLY, NNZ_MAX*sizeof(float), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_col_indices(context, CL_MEM_READ_ONLY, NNZ_MAX*sizeof(int), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_y_fpga(context, CL_MEM_WRITE_ONLY, ROW_SIZE_MAX*sizeof(float), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_row_ptr(context, CL_MEM_READ_ONLY, (ROW_SIZE_MAX+1)*sizeof(int), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_row_size(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_nnz(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err));
    
	y_gold      = (DATA_TYPE *)malloc(ROW_SIZE_MAX*sizeof(DATA_TYPE));

    //set the kernel Arguments
    int narg=0;
    OCL_CHECK(err, err = krnl_sparse_mv.setArg(narg++,buffer_row_ptr));
    OCL_CHECK(err, err = krnl_sparse_mv.setArg(narg++,buffer_col_indices));
    OCL_CHECK(err, err = krnl_sparse_mv.setArg(narg++,buffer_values));
    OCL_CHECK(err, err = krnl_sparse_mv.setArg(narg++,buffer_y_fpga));
    OCL_CHECK(err, err = krnl_sparse_mv.setArg(narg++,buffer_x));
    OCL_CHECK(err, err = krnl_sparse_mv.setArg(narg++,buffer_row_size));
    OCL_CHECK(err, err = krnl_sparse_mv.setArg(narg++,buffer_nnz));

    //We then need to map our OpenCL buffers to get the pointers
    OCL_CHECK(err, x = (float*)q.enqueueMapBuffer (buffer_x , CL_TRUE , CL_MAP_WRITE , 0, COL_SIZE_MAX*sizeof(float), NULL, NULL, &err));
    OCL_CHECK(err, values = (float*)q.enqueueMapBuffer (buffer_values , CL_TRUE , CL_MAP_WRITE , 0, NNZ_MAX*sizeof(float), NULL, NULL, &err));
    OCL_CHECK(err, col_indices = (int*)q.enqueueMapBuffer (buffer_col_indices , CL_TRUE , CL_MAP_WRITE , 0, NNZ_MAX*sizeof(int), NULL, NULL, &err));
    OCL_CHECK(err, row_ptr = (int*)q.enqueueMapBuffer (buffer_row_ptr , CL_TRUE , CL_MAP_WRITE , 0, (ROW_SIZE_MAX+1)*sizeof(int), NULL, NULL, &err));
    OCL_CHECK(err, y_fpga = (float*)q.enqueueMapBuffer (buffer_y_fpga , CL_TRUE , CL_MAP_READ , 0, (ROW_SIZE_MAX)*sizeof(float), NULL, NULL, &err));
    OCL_CHECK(err, row_size = (int*)q.enqueueMapBuffer (buffer_row_size , CL_TRUE , CL_MAP_WRITE , 0, sizeof(int), NULL, NULL, &err));
    OCL_CHECK(err, nnz = (int*)q.enqueueMapBuffer (buffer_nnz , CL_TRUE , CL_MAP_WRITE , 0, sizeof(int), NULL, NULL, &err));


    *row_size = 10;
    *nnz = 0;
    generateSpMV(row_size_val, col_size, nnz);
    printf("NNZ = %d\n", *nnz);

//    	read_mtx_SpMV(argv[1], &row_size, &col_size, &nnz);

	for (int i = 0; i < col_size; i++) {
		x[i] = (1.0*rand()+1.0)/RAND_MAX;
	}

	gold_spmv();

    // Data will be migrated to kernel space
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_x,buffer_values,buffer_col_indices,buffer_row_ptr},0/* 0 means from host*/));

    //Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_sparse_mv));

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_y_fpga},CL_MIGRATE_MEM_OBJECT_HOST));

    OCL_CHECK(err, q.finish());

    //Verify the result



	int status = 0;
	for(int i=0;i<row_size_val;i++) {
		DATA_TYPE diff = fabs(y_gold[i]-y_fpga[i]);
		if(diff > 0.1 || diff != diff){
			printf("error occurs at  %d with value y_hw = %lf, should be y_gold = %lf \n",i,y_fpga[i],y_gold[i]);
			status = -1;
			break;
		}
//		printf("Row %d: %f\n", i, y_fpga[i]);
	}

	if(!status) {
		printf("Hardware Validation PASSED!\n");
	} else {
		printf("Hardware Validation FAILED!\n");
		return -1;
	}


    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_x , x));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_values , values));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_col_indices , col_indices));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_row_ptr , row_ptr));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_y_fpga , y_fpga));
    OCL_CHECK(err, err = q.finish());

    return status;

}
