#include "spmv.h"
#include <hls_stream.h>

#define II 4
#define NUM_STREAMS 4

extern "C" {
void spmv_kernel(
		int rows_length[NUM_ROWS],
		int rows_length_pad[NUM_ROWS],
		int cols[NNZ],
		DTYPE values[NNZ],
		DTYPE y[SIZE],
		DTYPE x_dup[NUM_STREAMS][SIZE],
		int new_nnz)
{
#pragma HLS DATAFLOW


#pragma HLS ARRAY_PARTITION variable=rows_length_pad dim=1
#pragma HLS ARRAY_PARTITION variable=rows_length dim=1


	int row_length_pad[NUM_STREAMS] = {0}, row_length[NUM_STREAMS] = {0}, k[NUM_STREAMS] = {0}, row_counter[NUM_STREAMS] = {0};
	hls::stream<DTYPE> values_fifo[NUM_STREAMS];
	hls::stream<int>   cols_fifo[NUM_STREAMS];
	hls::stream<DTYPE> results_fifo[NUM_STREAMS];

#pragma HLS stream variable=values_fifo[0] type=fifo depth=32
#pragma HLS stream variable=values_fifo[1] type=fifo depth=32
#pragma HLS stream variable=values_fifo[2] type=fifo depth=32
#pragma HLS stream variable=values_fifo[3] type=fifo depth=32
#pragma HLS stream variable=cols_fifo[0] type=fifo depth=32
#pragma HLS stream variable=cols_fifo[1] type=fifo depth=32
#pragma HLS stream variable=cols_fifo[2] type=fifo depth=32
#pragma HLS stream variable=cols_fifo[3] type=fifo depth=32
#pragma HLS stream variable=results_fifo[0] type=fifo depth=32
#pragma HLS stream variable=results_fifo[1] type=fifo depth=32
#pragma HLS stream variable=results_fifo[2] type=fifo depth=32
#pragma HLS stream variable=results_fifo[3] type=fifo depth=32


//#pragma HLS INTERFACE mode=axis port=values_fifo[0]
//#pragma HLS INTERFACE mode=axis port=values_fifo[1]
//#pragma HLS INTERFACE mode=axis port=cols_fifo[0]
//#pragma HLS INTERFACE mode=axis port=cols_fifo[1]
//#pragma HLS INTERFACE mode=axis port=results_fifo[0]
//#pragma HLS INTERFACE mode=axis port=results_fifo[1]


	DTYPE sum[NUM_STREAMS] = {0};
	DTYPE value[NUM_STREAMS];
	int col[NUM_STREAMS];
	DTYPE term[NUM_STREAMS][II];
	int new_nnz_split[NUM_STREAMS] = {0};
	int nnz_split[NUM_STREAMS] = {0};

#pragma HLS partition variable=value dim=1
#pragma HLS partition variable=col dim=1
#pragma HLS partition variable=term dim=1


	int index[NUM_STREAMS] = {0};
	for(int i = 0; i < NUM_STREAMS; i++) {
		for (int l = 0; l < NUM_ROWS/NUM_STREAMS; l++) {
			new_nnz_split[i] += rows_length_pad[i*NUM_ROWS/NUM_STREAMS + l];
			nnz_split[i] += rows_length[i*NUM_ROWS/NUM_STREAMS + l];
		}
	}

	for (int i = 0; i < NUM_STREAMS - 1; i++) {
		index[i+1] = nnz_split[i]+index[i];
	}

	for(int i = 0; i < NUM_STREAMS; i++) {
			for(int j = 0; j < new_nnz_split[i]; j++) {
				if (j < nnz_split[i]) {
					values_fifo[i] << values[index[i]];
					cols_fifo[i] << cols[index[i]++];
				}
			}
	}

//	for (int i = 0; i < NNZ; i++) {
//#pragma HLS PIPELINE
//		values_fifo << values[i];
//		cols_fifo   << cols[i];
//	}
	for(int l = 0; l < NUM_STREAMS; l++) {
#pragma HLS unroll
//		for (int i = 0; i < new_nnz_split[l]; i+=II) {
		for (int i = 0; i < NUM_ROWS/NUM_STREAMS; i++) {
//#pragma HLS pipeline off
			for (int n = 0; n < rows_length_pad[l*NUM_ROWS/NUM_STREAMS + i]; n+=II) {

				#pragma HLS PIPELINE
				if (row_length_pad[l] == 0) {
					row_length_pad[l] = rows_length_pad[l*NUM_ROWS/NUM_STREAMS + i];
					row_length[l] = rows_length[l*NUM_ROWS/NUM_STREAMS + i];
					row_counter[l] = 0;
					sum[l] = 0;
				}

				for (int j = 0; j < II; j++) {
					row_counter[l]++;
					if (row_counter[l] > row_length[l]) {
						term[l][j] = 0;
					} else {
						value[l] = values_fifo[l].read();
						col[l]   = cols_fifo[l].read();
						term[l][j] = value[l] * x_dup[l][col[l]];
					}
				}

				DTYPE sum_tmp = 0;
				for (int j = 0; j < II; j++) {
					sum_tmp += term[l][j];
				}
				sum[l] += sum_tmp;

				row_length_pad[l] -= II;
				if (row_length_pad[l] == 0) {
					results_fifo[l] << sum[l];
				}
			}
		}
	}

	for(int l = 0; l < NUM_STREAMS; l++) {
//#pragma HLS unroll
		for (int i = 0; i < NUM_ROWS/NUM_STREAMS; i++) {
	#pragma HLS PIPELINE
			y[l*NUM_ROWS/NUM_STREAMS + i] = results_fifo[l].read();
		}
	}
}
void krnl_spmv_fast_multiport(
		int rowPtr[NUM_ROWS + 1],
		int cols[NNZ],
		DTYPE values[NNZ],
		DTYPE y[SIZE],
		DTYPE x[SIZE])
{

#pragma HLS INTERFACE mode=m_axi port=rowPtr offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=cols offset=slave bundle=gmem1
#pragma HLS INTERFACE mode=m_axi port=values offset=slave bundle=gmem2
#pragma HLS INTERFACE mode=m_axi port=x offset=slave bundle=gmem3

	// rowPtr to rows_length
	int rows_length[NUM_ROWS] = {0};
//	int rows_length_int[NUM_ROWS] = {0};
	for (int i = 1; i < NUM_ROWS + 1; i++) {
#pragma HLS PIPELINE off
		rows_length[i - 1] = rowPtr[i] - rowPtr[i - 1];
//		rows_length_int[i-1] = rowPtr[i] - rowPtr[i-1];
	}

	int rows_length_pad[NUM_ROWS];


	int new_nnz = 0;
	for (int i = 0; i < NUM_ROWS; i++) {
#pragma HLS PIPELINE off
		int r = rows_length[i];
		int r_diff = r % II;
		if (r == 0) {
			rows_length_pad[i] = II;
			new_nnz += II;
		} else if (r_diff != 0) {
			rows_length_pad[i] = r + (II - r_diff);
			new_nnz += r + (II - r_diff);
		} else {
			rows_length_pad[i] = r;
			new_nnz += r;
		}
	}


	DTYPE x_dup[NUM_STREAMS][SIZE];

	for (int i = 0; i < NUM_STREAMS; i++) {
		for (int k=0; k < SIZE; k++) {
			x_dup[i][k] = x[k];
		}
	}

	spmv_kernel(rows_length, rows_length_pad, cols, values, y, x_dup, new_nnz);

}
}
