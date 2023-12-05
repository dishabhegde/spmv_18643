#ifndef __SPMV_H__
#define __SPMV_H__

const static int SIZE = 256; // SIZE of square matrix
const static int NNZ = 6500; //Number of non-zero elements
const static int NUM_ROWS = 256;// SIZE;
typedef float DTYPE;
extern "C" void krnl_spmv_fast_multiport(
		int rows_length[NUM_ROWS],
		int rows_length_pad[NUM_ROWS],
		int cols[NNZ],
		DTYPE values[NNZ],
		DTYPE y[SIZE],
		DTYPE x[SIZE]);
extern "C" void krnl_spmv_fast(
		int rowPtr[NUM_ROWS + 1],
		int cols[NNZ],
		DTYPE values[NNZ],
		DTYPE y[SIZE],
		DTYPE x[SIZE]);

#endif // __MATRIXMUL_H__ not defined
