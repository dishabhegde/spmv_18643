#include "spmv.h"
#include <hls_stream.h>

#define II 7
#define NUM_STREAMS 2

extern "C" {
void krnl_spmv_fast_multiport(
		const int rowPtr[NUM_ROWS + 1],
		const int cols[NNZ],
		const DTYPE values[NNZ],
		DTYPE y[SIZE],
		const DTYPE x[SIZE])
{

	// rowPtr to rows_length
	int rows_length[NUM_ROWS] = {0};
	for (int i = 1; i < NUM_ROWS + 1; i++) {
#pragma HLS PIPELINE
		rows_length[i - 1] = rowPtr[i] - rowPtr[i - 1];
	}

	int rows_length_pad[NUM_ROWS];
	int new_nnz = 0;
	for (int i = 0; i < NUM_ROWS; i++) {
#pragma HLS PIPELINE
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

#pragma HLS DATAFLOW

	int row_length_pad[NUM_STREAMS] = {0}, row_length[NUM_STREAMS] = {0}, k[NUM_STREAMS] = {0}, row_counter[NUM_STREAMS] = {0};
	hls::stream<DTYPE> values_fifo[NUM_STREAMS];
	hls::stream<int>   cols_fifo[NUM_STREAMS];
	hls::stream<DTYPE> results_fifo[NUM_STREAMS];

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
		for (int i = 0; i < new_nnz_split[l]; i+=II) {
		#pragma HLS PIPELINE
			if (row_length_pad[l] == 0) {
				row_length_pad[l] = rows_length_pad[l*NUM_ROWS/NUM_STREAMS + k[l]];
				row_length[l] = rows_length[l*NUM_ROWS/NUM_STREAMS + k[l]++];
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

	for(int l = 0; l < NUM_STREAMS; l++) {
		for (int i = 0; i < NUM_ROWS/NUM_STREAMS; i++) {
	#pragma HLS PIPELINE
			y[l*NUM_ROWS/NUM_STREAMS + i] = results_fifo[l].read();
		}
	}
}
}
