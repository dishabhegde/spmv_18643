#include "spmv.h"
#include <hls_stream.h>

#define II 7
#define NUM_STREAMS 2

extern "C" {
void krnl_spmv_fast_multiport(
		int rows_length[NUM_ROWS],
		int rows_length_pad[NUM_ROWS],
		int cols[NNZ],
		DTYPE values[NNZ],
		DTYPE y[SIZE],
		DTYPE x[SIZE])
{
#pragma HLS DATAFLOW

#pragma HLS ARRAY_PARTITION variable=rows_length_pad dim=1
#pragma HLS ARRAY_PARTITION variable=x dim=1

	int row_length_pad[NUM_STREAMS] = {0}, row_length[NUM_STREAMS] = {0}, k[NUM_STREAMS] = {0}, row_counter[NUM_STREAMS] = {0};
	hls::stream<DTYPE> values_fifo[NUM_STREAMS];
	hls::stream<int>   cols_fifo[NUM_STREAMS];
	hls::stream<DTYPE> results_fifo[NUM_STREAMS];

#pragma HLS stream variable=values_fifo[0] type=fifo depth=32
#pragma HLS stream variable=values_fifo[1] type=fifo depth=32
#pragma HLS stream variable=cols_fifo[0] type=fifo depth=32
#pragma HLS stream variable=cols_fifo[1] type=fifo depth=32
#pragma HLS stream variable=results_fifo[0] type=fifo depth=32
#pragma HLS stream variable=results_fifo[1] type=fifo depth=32

//
#pragma HLS INTERFACE mode=axis port=values_fifo[0]
#pragma HLS INTERFACE mode=axis port=values_fifo[1]
#pragma HLS INTERFACE mode=axis port=cols_fifo[0]
#pragma HLS INTERFACE mode=axis port=cols_fifo[1]
#pragma HLS INTERFACE mode=axis port=results_fifo[0]
#pragma HLS INTERFACE mode=axis port=results_fifo[1]


	DTYPE sum[NUM_STREAMS] = {0};
	DTYPE value[NUM_STREAMS];
	int col[NUM_STREAMS];
	DTYPE term[NUM_STREAMS][II];
	int new_nnz_split[NUM_STREAMS] = {0};
	int nnz_split[NUM_STREAMS] = {0};

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
//		for (int i = 0; i < new_nnz_split[l]; i+=II) {
		for (int i = 0; i < NUM_ROWS/NUM_STREAMS; i++) {
#pragma HLS pipeline off
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
						term[l][j] = value[l] * x[col[l]];
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
		for (int i = 0; i < NUM_ROWS/NUM_STREAMS; i++) {
	#pragma HLS PIPELINE
			y[l*NUM_ROWS/NUM_STREAMS + i] = results_fifo[l].read();
		}
	}
}

}
