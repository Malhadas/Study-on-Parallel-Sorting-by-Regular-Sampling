#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>


//auxiliary function for merge_sort
double *merge(double * left, double * right, int l_end, int r_end) {
	int temp_off, l_off, r_off, size = l_end+r_end;
	double *temp = (double *)malloc(sizeof(double) * l_end);

	// Copy lower half into temp buffer
	for(l_off=0, temp_off=0; left+l_off != right; l_off++, temp_off++){
		*(temp + temp_off) = *(left + l_off);
	}
	
	temp_off=0; l_off=0; r_off=0;
	
	while(l_off < size){
		if(temp_off < l_end){
			if(r_off < r_end){
				if(*(temp+temp_off) < *(right+r_off)){
					*(left+l_off) = *(temp+temp_off);
					temp_off++;
				}
				else {
					*(left+l_off) = *(right+r_off);
					r_off++;
				}
			}
			else {
				*(left+l_off) = *(temp+temp_off);
				temp_off++;
			}
		}
		else {
			if(r_off < r_end) {
				*(left + l_off) = *(right + r_off);
				r_off++;
			}
			else {
				printf("\nERROR - merging loop going too far\n");
			}
		}

		l_off++;
	}
	free(temp);
	return left;
}

// sort array 'arr' first 'size' elements with sequential merge sort
double *merge_sort(double * arr, int size) {
// Arrays shorter than 1 are already sorted
	if(size > 1){
		int middle = size / 2, i;
		double *left, *right;

		left = arr;
		right = arr + middle;
		left = merge_sort(left, middle);
		right = merge_sort(right, size-middle);
		return merge(left, right, middle,size-middle);
	}
	else { 
		return arr; 
	}
}

//compare double
int compare(const void * ptr2num1, const void * ptr2num2) {
	double num1 = *((double*) ptr2num1);
	double num2 = *((double*) ptr2num2);

	if ( num1 > num2 )
		return 1;
	else if ( num1 < num2 )
		return -1;
	else
		return 0;
}

//sort array 'a'first 'len' elements using quick sort
void sortdd(double *a, int len) {
	qsort(a, len, sizeof(double), compare);
}

// determine the boundaries for the sublists of a local array
void calc_partition_borders(double array[], int start, int end, int result[], int at, double pivots[], int first_pv, int last_pv) {
	int mid, lowerbound, upperbound, center;
	double pv;

	mid = (first_pv + last_pv) / 2;
	pv = pivots[mid-1];
	lowerbound = start;
	upperbound = end;

	while(lowerbound <= upperbound) {
		center = (lowerbound + upperbound) / 2;
		if(array[center] > pv) {
			upperbound = center - 1;
		} 
		else {
			lowerbound = center + 1;
		}
	}

	result[at + mid] = lowerbound;
	
	if(first_pv < mid) {
		calc_partition_borders(array, start, lowerbound - 1, result, at, pivots, first_pv, mid - 1);
	}

	if(mid < last_pv) {
		calc_partition_borders(array, lowerbound, end, result, at, pivots, mid + 1, last_pv);
	}
}

/* sort an array */
void psrs_sort(double *a, int n, int p) {
	if (n > 1){
				int size, num, rsize, sample_size;
				double *sample, *pivots;
				int *partition_borders, *bucket_sizes, *result_positions;
				double **loc_a_ptrs;

				// Determine the appropriate number of threads to use
				// p^3 <= n
				num = omp_get_max_threads();
                if (p>num) p = num;
				p = p*p*p;
				if (p > n){
					p = floor(pow(n,0.33));
					p-=p%2;
				}
				else{
					num = omp_get_max_threads();
                	if (p>num) p = num;
					p-=p%2;
					if (p<=0) p = 1;
				}

				omp_set_num_threads(p);
				printf("%d ",p);
				size = (n + p - 1) / p;
				rsize = (size + p - 1) / p;
				sample_size = p * (p - 1);
				loc_a_ptrs = (double **)malloc(p * sizeof(double *));
				sample = (double *)malloc(sample_size * sizeof(double));
				partition_borders = (int*)malloc(p * (p + 1) * sizeof(int));
				bucket_sizes = (int*)malloc(p * sizeof(int));
				result_positions = (int*)malloc(p * sizeof(int));
				pivots = (double *)malloc((p - 1) * sizeof(double));
				
				#pragma omp parallel
				{
				
				int i, j, max, thread_num, start, end, loc_size, offset, this_result_size;
				double *loc_a, *this_result, *current_a;
				thread_num = omp_get_thread_num();
				start = thread_num * size;
				end = start + size - 1;
				
				if(end >= n) end = n - 1;
	
				loc_size = (end - start + 1);
				end = end % size;
				loc_a = (double *)malloc(loc_size * sizeof(double));
				memcpy(loc_a, a + start, loc_size * sizeof(double));
				loc_a_ptrs[thread_num] = loc_a;
				sortdd(loc_a, loc_size);
				offset = thread_num * (p - 1) - 1;
				for(i = 1; i < p; i++) {
					if(i * rsize <= end) {
						sample[offset + i] = loc_a[i * rsize - 1];
					} 
					else {
						sample[offset + i] = loc_a[end];
					}
				}

				#pragma omp barrier
				
				#pragma omp single
				{
					merge_sort(sample, sample_size);
					for(i = 0; i < p - 1; i++) {
						pivots[i] = sample[i * p + p / 2];
					}
				}
				
				#pragma omp barrier

				offset = thread_num * (p + 1);
				partition_borders[offset] = 0;
				partition_borders[offset + p] = end + 1;
				calc_partition_borders(loc_a, 0, loc_size-1, partition_borders, offset, pivots, 1, p-1);
			
				#pragma omp barrier
	
				max = p * (p + 1);
				bucket_sizes[thread_num] = 0;
				for(i = thread_num; i < max; i += p + 1) {
					bucket_sizes[thread_num] += partition_borders[i + 1] - partition_borders[i];
				}

				#pragma omp barrier
				
				#pragma omp single
				{
					result_positions[0] = 0;
					for(i = 1; i < p; i++) {
						result_positions[i] = bucket_sizes[i-1] + result_positions[i-1];
					}
				}

				#pragma omp barrier

				this_result = a + result_positions[thread_num];
				if(thread_num == p-1) {
					this_result_size = n - result_positions[thread_num];
				} 
				else {
					this_result_size = result_positions[thread_num+1] - result_positions[thread_num];
				}

				// pluck this threads sublist from each of the local arrays
				this_result = a + result_positions[thread_num];
				
				for(i = 0, j = 0; i < p; i++) {
					int low, high, partition_size;
					offset = i * (p + 1) + thread_num;
					low = partition_borders[offset];
					high = partition_borders[offset+1];
					partition_size = (high - low);

					if(partition_size > 0) {
						memcpy(this_result+j, &(loc_a_ptrs[i][low]), partition_size * sizeof(double));
						j += partition_size;
					}
				}

				// sort p local sorted arrays
				sortdd(this_result, this_result_size);

				#pragma omp barrier

				free(loc_a);
			}

			free(loc_a_ptrs);
			free(sample);
			free(partition_borders);
			free(bucket_sizes);
			free(result_positions);
			free(pivots);
		//}
	}
}

double * copy_array(double * array, int size) {

	double * result = (double *)malloc(sizeof(double) * size);
	int i;

	for ( i = 0; i < size; i++ ) {
		result[i] = array[i];
	}
	return result;
}

/* create an array of length size and fill it with random numbers */
double * gen_random(int size) {

	double * result = (double *)malloc(sizeof(double) * size);
	int i;
	struct timeval seedtime;
	int seed;
	
	srand(time(NULL));

	/* fill the array with random numbers */
	for ( i = 0; i < size; i++ ) {
		result[i] = (double)rand() / (double)RAND_MAX;
	}

	return result;
}

int main(int argc, char ** argv) {
	double * array;
	double * copy;
	double sort_time;
	int array_size;
	struct timeval start_time;
	struct timeval stop_time;
	int i;

	if ( argc == 3) { /* generate random data */
			array_size = atoi(argv[1]);
			array = gen_random(array_size);
	}
	else {
		printf("Usage: ./<program_name> <size> <thread_num>\n");
		exit(1);
	}

	copy = copy_array(array, array_size);

	gettimeofday(&start_time, NULL);
	psrs_sort(array, array_size, atoi(argv[2]));
	gettimeofday(&stop_time, NULL);
	
	printf("Took %lld ms\n", (long long) (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
(stop_time.tv_usec - start_time.tv_usec));

	/* sort the copy. using qsort */
	qsort(copy, array_size, sizeof(double), compare);

	/* now check that the two are identical */
	for ( i = 0; i < array_size; i++ ) {
		if ( array[i] != copy[i] ) {
			printf("Error at position %d. Found %g, should be %g.\n", i, array[i], copy[i]);
		}
	}

	return 0;
}



