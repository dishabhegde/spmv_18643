#include "spmv.h"
#include <hls_stream.h>

#define II 2
#define NUM_STREAMS 2

extern "C" {
void spmv_kernel(
		volatile int rows_length[NUM_ROWS],
		volatile int rows_length_pad[NUM_ROWS],
		int cols[NNZ],
		DTYPE values[NNZ],
		DTYPE y[SIZE],
		DTYPE x_dup[NUM_STREAMS][SIZE])
{
#pragma HLS DATAFLOW


#pragma HLS ARRAY_PARTITION variable=rows_length_pad dim=1
#pragma HLS ARRAY_PARTITION variable=rows_length dim=1
#pragma HLS ARRAY_PARTITION variable=x_dup dim=1


	int row_length_pad[NUM_STREAMS] = {0}, row_length[NUM_STREAMS] = {0}, row_counter[NUM_STREAMS] = {0};

#pragma HLS ARRAY_PARTITION variable=row_length_pad dim=1
#pragma HLS ARRAY_PARTITION variable=row_length dim=1
#pragma HLS ARRAY_PARTITION variable=row_counter dim=1

	hls::stream<DTYPE> values_fifo[NUM_STREAMS];
	hls::stream<int>   cols_fifo[NUM_STREAMS];
	hls::stream<DTYPE> results_fifo[NUM_STREAMS];

#pragma HLS stream variable=values_fifo[0] type=fifo depth=32
#pragma HLS stream variable=values_fifo[1] type=fifo depth=32
//#pragma HLS stream variable=values_fifo[2] type=fifo depth=32
//#pragma HLS stream variable=values_fifo[3] type=fifo depth=32
//#pragma HLS stream variable=values_fifo[4] type=fifo depth=32
//#pragma HLS stream variable=values_fifo[5] type=fifo depth=32
//#pragma HLS stream variable=values_fifo[6] type=fifo depth=32
//#pragma HLS stream variable=values_fifo[7] type=fifo depth=32
#pragma HLS stream variable=cols_fifo[0] type=fifo depth=32
#pragma HLS stream variable=cols_fifo[1] type=fifo depth=32
//#pragma HLS stream variable=cols_fifo[2] type=fifo depth=32
//#pragma HLS stream variable=cols_fifo[3] type=fifo depth=32
//#pragma HLS stream variable=cols_fifo[4] type=fifo depth=32
//#pragma HLS stream variable=cols_fifo[5] type=fifo depth=32
//#pragma HLS stream variable=cols_fifo[6] type=fifo depth=32
//#pragma HLS stream variable=cols_fifo[7] type=fifo depth=32
#pragma HLS stream variable=results_fifo[0] type=fifo depth=32
#pragma HLS stream variable=results_fifo[1] type=fifo depth=32
//#pragma HLS stream variable=results_fifo[2] type=fifo depth=32
//#pragma HLS stream variable=results_fifo[3] type=fifo depth=32
//#pragma HLS stream variable=results_fifo[4] type=fifo depth=32
//#pragma HLS stream variable=results_fifo[5] type=fifo depth=32
//#pragma HLS stream variable=results_fifo[6] type=fifo depth=32
//#pragma HLS stream variable=results_fifo[7] type=fifo depth=32


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

#pragma HLS ARRAY_PARTITION variable=value dim=1
#pragma HLS ARRAY_PARTITION variable=col dim=1
#pragma HLS ARRAY_PARTITION variable=term dim=1
#pragma HLS ARRAY_PARTITION variable=sum dim=1
#pragma HLS ARRAY_PARTITION variable=new_nnz_split dim=1
#pragma HLS ARRAY_PARTITION variable=nnz_split dim=1


	int index[NUM_STREAMS] = {0};
	Loop_nnz: for(int i = 0; i < NUM_STREAMS; i++) {
#pragma HLS unroll factor=2
		for (int l = 0; l < NUM_ROWS/NUM_STREAMS; l++) {
#pragma pipeline off
			new_nnz_split[i] += rows_length_pad[i*NUM_ROWS/NUM_STREAMS + l];
			nnz_split[i] += rows_length[i*NUM_ROWS/NUM_STREAMS + l];
		}
	}

	Loop_index: for (int i = 0; i < NUM_STREAMS - 1; i++) {
		index[i+1] = nnz_split[i]+index[i];
	}

	Loop_fifo_fill: for(int i = 0; i < NUM_STREAMS; i++) {
#pragma HLS unroll factor=2
			for(int j = 0; j < new_nnz_split[i]; j++) {
#pragma HLS pipeline off
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
	Loop_streams: for(int l = 0; l < NUM_STREAMS; l++) {
#pragma HLS unroll factor=2
		Loop_rows: for (int i = 0; i < NUM_ROWS/NUM_STREAMS; i++) {
			Loop_onerow: for (int n = 0; n < rows_length_pad[l*NUM_ROWS/NUM_STREAMS + i]; n+=II) {

				#pragma HLS PIPELINE
				if (row_length_pad[l] == 0) {
					row_length_pad[l] = rows_length_pad[l*NUM_ROWS/NUM_STREAMS + i];
					row_length[l] = rows_length[l*NUM_ROWS/NUM_STREAMS + i];
					row_counter[l] = 0;
					sum[l] = 0;
				}

				Loop_II_sec: for (int j = 0; j < II; j++) {
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

	Loop_result: for(int l = 0; l < NUM_STREAMS; l++) {
//#pragma HLS unroll
		for (int i = 0; i < NUM_ROWS/NUM_STREAMS; i++) {
	#pragma HLS PIPELINE off
			y[l*NUM_ROWS/NUM_STREAMS + i] = results_fifo[l].read();
		}
	}
}
void krnl_spmv_fast_multiport(
		int cols[NNZ],
		DTYPE values[NNZ],
		DTYPE x[SIZE],
		int rows_length_in[NUM_ROWS],
		int rows_length_pad_in[NUM_ROWS],
		DTYPE y[SIZE])
{

//#pragma HLS INTERFACE mode=bram port=rows_length
//#pragma HLS INTERFACE mode=bram port=cols
//#pragma HLS INTERFACE mode=bram port=values
//#pragma HLS INTERFACE mode=bram port=rows_length_pad

	// rowPtr to rows_length
	volatile int rows_length[NUM_ROWS];
//	int rows_length_int[NUM_ROWS] = {0};
//	for (int i = 1; i < NUM_ROWS + 1; i++) {
//#pragma HLS PIPELINE off
//		rows_length[i - 1] = rowPtr[i] - rowPtr[i - 1];
////		rows_length_int[i-1] = rowPtr[i] - rowPtr[i-1];
//	}
//
	volatile int rows_length_pad[NUM_ROWS];

#pragma HLS ARRAY_PARTITION variable=rows_length_pad dim=1
#pragma HLS ARRAY_PARTITION variable=rows_length dim=1
//
//
//	int new_nnz = 0;
//	for (int i = 0; i < NUM_ROWS; i++) {
//#pragma HLS PIPELINE
//		int r = rows_length[i];
//		int r_diff = r % II;
//		if (r == 0) {
//			rows_length_pad[i] = II;
//			new_nnz += II;
//		} else if (r_diff != 0) {
//			rows_length_pad[i] = r + (II - r_diff);
//			new_nnz += r + (II - r_diff);
//		} else {
//			rows_length_pad[i] = r;
//			new_nnz += r;
//		}
//	}

	for (int i = 0; i < NUM_ROWS; i++) {
		rows_length[i] = rows_length_in[i];
		rows_length_pad[i] = rows_length_pad_in[i];

	}
	DTYPE x_dup[NUM_STREAMS][SIZE];

#pragma HLS ARRAY_PARTITION variable=x_dup dim=1
	for (int i = 0; i < NUM_STREAMS; i++) {
		for (int k=0; k < SIZE; k++) {

			x_dup[i][k] = x[k];
		}
	}

	spmv_kernel(rows_length, rows_length_pad, cols, values, y, x_dup);

}
}
