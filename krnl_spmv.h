/* File: spmv_mohammad.h
 *
 Copyright (c) [2016] [Mohammad Hosseinabady (mohammad@hosseinabady.com)]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===============================================================================
*
* File name : spmv_mohammad.h
* Author    : Mohammad Hosseinabady mohammad@hosseinabady.com
* Date      : 14 September 2018
* Blog      : https://highlevel-synthesis.com/
* Project   : ENEAC project at Bristol University (Jose Nunez-Yanez)
*/

//#ifndef __SPMV_MOHAMMAD__
//#define __SPMV_MOHAMMAD__



const static int II = 9;

typedef float DATA_TYPE;
const static int ROW_SIZE_MAX = (1024*128);
const static int COL_SIZE_MAX = ROW_SIZE_MAX;
const static int NNZ_MAX =  (1024*1024*16);
const static int NUM_ROWS = 10;
extern "C" {
void spmv_mohammad(
		int *rowPtr,
		int *columnIndex,
		DATA_TYPE *values,
		DATA_TYPE *y,
		DATA_TYPE *x,
		int *row_size,
		int *nnz);


void spmv_openmp(
		DATA_TYPE *z,
		DATA_TYPE *data,
		int       *colind,
		int       *row_ptr,
		DATA_TYPE *x,
		int        row_size,
		int        col_size,
		int        data_size);
//#endif
}
