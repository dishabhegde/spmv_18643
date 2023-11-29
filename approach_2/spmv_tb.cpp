#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

#include <stdlib.h>
#include <stdio.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "spmv_tb.h"

#include "time.h"
#include <sys/time.h>

using namespace std;

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

DTYPE M[SIZE][SIZE] = {0};
DTYPE x[SIZE] = {0};
DTYPE values[NNZ] = {0};
int columnIndex[NNZ] = {0};
int rowPtr[NUM_ROWS+1] = {0};
DTYPE y_sw[SIZE] = {0};
//DTYPE y[SIZE] = {0};

void matrixvector(DTYPE A[SIZE][SIZE], DTYPE *y, DTYPE *x)
{
	for (int i = 0; i < SIZE; i++) {
		DTYPE y0 = 0;
		for (int j = 0; j < SIZE; j++)
			y0 += A[i][j] * x[j];
		y[i] = y0;
	}
}

void load_matrix()
{
	ifstream matrix("./matrix.dat");
	string line;
	int tmp, i = 0, j = 0;
	while (getline(matrix, line)) {
		istringstream ss(line);
		while (ss >> tmp) {
			if (tmp > 0) {
				M[i][j] = tmp;
			}
			j++;
		}
		i++;
		j = 0;
	}
	matrix.close();
//	FILE *input = fopen("matrix.dat", "r");
//	for(int i = 0; i < SIZE; i++){
//		fgets((char*)M[i], SIZE * sizeof(DTYPE), input);
//	}
//	fclose(input);
}

void load_data()
{
//	std::cout << "Start loading data..." << std:: endl;
//	ifstream data("/scratch/643_vitis_zhengyu3/spmv/spmv/src/data.dat");
//	string line;
//	int tmp, i = 0;
//
//	getline(data, line);
//	istringstream ss(line);
//	while (ss >> tmp) {
//		values[i++] = tmp;
//		std::cout << "value " << tmp << " stored" << std:: endl;
//	}
//
//	data.close();

//	std::cout << filesystem::current_path() << std::endl;
//	FILE *input = fopen("/scratch/643_vitis_zhengyu3/spmv/spmv/src/data.dat", "rb");
	FILE *input = fopen("./data.dat", "rb");
	if (input == NULL)
		perror("Error opening the data file");
	fread(values, sizeof(DTYPE), sizeof(values), input);
//	fgets((char*)values, NNZ * sizeof(DTYPE), input);
	fclose(input);
}

void load_rows(int *ptr_rowPtr)
{
	ifstream rows("./rows.dat");
	string line;
	int tmp, i = 0;

	getline(rows, line);
	istringstream ss(line);
	while (ss >> tmp) {
		ptr_rowPtr[i++] = tmp;
	}

	rows.close();
}

void load_cols(int *ptr_columnIndex)
{
	ifstream cols("./cols.dat");
	string line;
	int tmp, i = 0;

	getline(cols, line);
	istringstream ss(line);
	while (ss >> tmp) {
		ptr_columnIndex[i++] = tmp;
	}

	cols.close();
}

void gen_input()
{
	for (int i = 0; i < SIZE; i++) {
		x[i] = rand() % 100;
	}
}

void generateSparseMatrix(int numRows, int numCols, double density) {

    // Initialize random number generator
    srand(time(NULL));

    int index = 0;
    rowPtr[0] = 0;

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols && index < NNZ; ++j) {
            double randValue = (double)rand() / RAND_MAX;
            if (randValue < density) {
                // Add a non-zero element to the matrix
                values[index] = rand() % 100;  // For simplicity, using random integers between 0 and 9
                columnIndex[index] = j;
                M[i][j] = values[index];
                index++;
            } else {
                M[i][j] = 0;
            }
        }
        rowPtr[i + 1] = index;
    }
}

double getTimestamp() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_usec + tv.tv_sec*1e6;
}

int main(int argc, char* argv[]) {

    //TARGET_DEVICE macro needs to be passed from gcc command line
    if(argc != 2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}

    std::string xclbinFilename = argv[1];

    // Creates a vector of DATA_SIZE elements with an initial value of 10 and 32
    // using customized allocator for getting buffer alignment to 4k boundary

    std::vector<cl::Device> devices;
    cl::Device device;
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel krnl_sparse_matrix_mul;
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
    std::cout << "Checkpoint 1" << std::endl;
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    std::cout << "Checkpoint 2" << std::endl;
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
    OCL_CHECK(err, krnl_sparse_matrix_mul = cl::Kernel(program,"krnl_spmv_fast", &err));

    std::cout << "Allocating the device buffers..." << std::endl;
    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device.
//    DTYPE *values = (DTYPE*) malloc(NNZ * sizeof(DTYPE));

    OCL_CHECK(err, cl::Buffer buffer_x(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, SIZE * sizeof(DTYPE), x, &err));
    OCL_CHECK(err, cl::Buffer buffer_values(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, NNZ * sizeof(DTYPE), values, &err));
    OCL_CHECK(err, cl::Buffer buffer_columnIndex(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, NNZ * sizeof(int), columnIndex, &err));
    OCL_CHECK(err, cl::Buffer buffer_rowPtr(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, (NUM_ROWS+1) * sizeof(int), rowPtr, &err));
    OCL_CHECK(err, cl::Buffer buffer_y(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(DTYPE), NULL, &err));

    //set the kernel Arguments
    int narg=0;
    OCL_CHECK(err, err = krnl_sparse_matrix_mul.setArg(narg++,buffer_rowPtr));
    OCL_CHECK(err, err = krnl_sparse_matrix_mul.setArg(narg++,buffer_columnIndex));
    OCL_CHECK(err, err = krnl_sparse_matrix_mul.setArg(narg++,buffer_values));
    OCL_CHECK(err, err = krnl_sparse_matrix_mul.setArg(narg++,buffer_y));
    OCL_CHECK(err, err = krnl_sparse_matrix_mul.setArg(narg++,buffer_x));

    //We then need to map our OpenCL buffers to get the pointers
//    int *ptr_rowPtr;
//    int *ptr_columnIndex;
//    DTYPE *ptr_values;
    DTYPE *ptr_y;
//    DTYPE *ptr_x;

    std::cout << "Mapping the device buffers..." << std::endl;

//    OCL_CHECK(err, ptr_rowPtr = (int*)q.enqueueMapBuffer (buffer_rowPtr , CL_TRUE , CL_MAP_WRITE , 0, (NUM_ROWS+1) * sizeof(int), NULL, NULL, &err));
//    OCL_CHECK(err, ptr_columnIndex = (int*)q.enqueueMapBuffer (buffer_columnIndex , CL_TRUE , CL_MAP_WRITE , 0, NNZ * sizeof(int), NULL, NULL, &err));
//    OCL_CHECK(err, ptr_values = (DTYPE*)q.enqueueMapBuffer (buffer_values , CL_TRUE , CL_MAP_WRITE , 0, NNZ * sizeof(DTYPE), NULL, NULL, &err));
//    OCL_CHECK(err, ptr_x = (DTYPE*)q.enqueueMapBuffer (buffer_x , CL_TRUE , CL_MAP_WRITE , 0, SIZE * sizeof(DTYPE), NULL, NULL, &err));
    OCL_CHECK(err, ptr_y = (DTYPE*)q.enqueueMapBuffer (buffer_y , CL_TRUE , CL_MAP_READ , 0, SIZE * sizeof(DTYPE), NULL, NULL, &err));

    std::cout << "Initializing the local buffers..." << std::endl;

    int fail = 0;
//	load_matrix();
//	load_data();
//	load_rows(ptr_rowPtr);
//	load_cols(ptr_columnIndex);
    generateSparseMatrix(SIZE, SIZE, (float)NNZ / (SIZE * SIZE));
	gen_input();

	//Test if the fed data is correct
	std::cout << "The matrix is:" << std::endl;
	for(int i = 0; i < SIZE; i++){
		for(int j = 0; j < SIZE; j++){
			std::cout << M[i][j] << " " ;
		}
		std::cout << std::endl;
	}
	std::cout << "The row pointers are:" << std::endl;
	for(int i = 0; i < NUM_ROWS + 1; i++){
		std::cout << rowPtr[i] << " " ;
	}
	std::cout << std::endl;
	std::cout << "The values are:" << std::endl;
	for(int i = 0; i < NNZ; i++){
		std::cout << values[i] << " " ;
	}
	std::cout << std::endl;
	std::cout << "The column indices are:" << std::endl;
	for(int i = 0; i < NNZ; i++){
		std::cout << columnIndex[i] << " " ;
	}
	std::cout << std::endl;
	std::cout << "The vector is:" << std::endl;
	for(int i = 0; i < SIZE; i++){
		std::cout << x[i] << " " ;
	}
	std::cout << std::endl;

	double hardware_start;
	double hardware_end;
	double hardware_execution_time;
	printf("\rHardware version started!\n\r");
	hardware_start = getTimestamp();

    // Data will be migrated to kernel space
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_rowPtr,buffer_columnIndex,buffer_values,buffer_x},0/* 0 means from host*/));

    std::cout << "Launching the kernel..." << std::endl;
    //Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_sparse_matrix_mul));

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_y},CL_MIGRATE_MEM_OBJECT_HOST));

    OCL_CHECK(err, q.finish());

    hardware_end = getTimestamp();
	printf("\rHardware version finished!\n\r");
	hardware_execution_time = (hardware_end-hardware_start)/(1000);
	printf("Hardware execution time  %.6f ms elapsed\n", hardware_execution_time);

    std::cout << "Finish the kernel running. Start on result checking..." << std::endl;

    matrixvector(M, y_sw, x);

//    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_rowPtr , ptr_rowPtr));
//    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_columnIndex , ptr_columnIndex));
//    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_values , ptr_values));
//    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_x , ptr_x));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_y , ptr_y));
    OCL_CHECK(err, err = q.finish());

    std::cout << "Software outputs are:" << std::endl;
	for(int i = 0; i < SIZE; i++){
		std::cout << y_sw[i] << " " ;
	}
	std::cout << std::endl;

	std::cout << "Kernel values are:" << std::endl;
	for(int i = 0; i < SIZE; i++){
		std::cout << ptr_y[i] << " " ;
	}
	std::cout << std::endl;

	for(int i = 0; i < SIZE; i++) {
		if(y_sw[i] != ptr_y[i]){
			std::cout << "Mismatching pair: " << y_sw[i] << ", " << ptr_y[i] << std::endl;
			fail = 1;
		}
	}

    std::cout << "TEST " << (fail ? "FAILED" : "PASSED") << std::endl;
    return (fail ? EXIT_FAILURE : EXIT_SUCCESS);

}
