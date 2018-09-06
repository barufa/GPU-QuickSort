#include <iostream>
#include <cstdio>
#include <algorithm>
#include <helper_cuda.h>
#include <helper_string.h>

#define MAX_DEPTH       16
#define INSERTION_SORT  32
#define SIZE            1000

__device__ void selection_sort(unsigned int *data, int left, int right){
    for(int i = left ; i <= right ; ++i){//Para cada i en [left,right]
        unsigned min_val = data[i];
        int min_idx = i;
        
        for (int j = i+1 ; j <= right ; ++j){//Encontrar el valor mas chico en [i, right]. 
            unsigned val_j = data[j];
            if (val_j < min_val){
                min_idx = j;
                min_val = val_j;
            }
        }
        
        if (i != min_idx){//Swapear min_idx con i
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

__global__ void simple_quicksort(unsigned int *data, int left, int right, int depth){
    if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT){//Si hay pocos elementos o estas muy profundo usar selection_sort
        selection_sort(data, left, right);
        return;
    }

    unsigned int *lptr = data+left;
    unsigned int *rptr = data+right;
    unsigned int  pivot = data[(left+right)/2];

    while (lptr <= rptr){//Ordenar parcialmente respecto del pivote
        unsigned int lval = *lptr;
        unsigned int rval = *rptr;

        while (lval < pivot){//Mover el puntero de la izquierda hasta encontrar un elemento menor que el pivote
            lptr++;
            lval = *lptr;
        }

        while (rval > pivot){//Mover el puntero de la derecha hasta encontrar un elemento mayor que el pivote
            rptr--;
            rval = *rptr;
        }

        if (lptr <= rptr){//Si los valores son validos, hacer el swap entre ellos
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    int nright = rptr - data;
    int nleft  = lptr - data;

    //Lanzar un nuevo bloque para ordenar la parte izquierda
    if (left < (rptr-data)){
        cudaStream_t s;//Definicion del objeto stream para hacer la concurrencia en el kernel
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);//Encolar la tarea en s
        cudaStreamDestroy(s);
    }

    //Lanzar un nuevo bloque para ordenar la parte derecha
    if ((lptr-data) < right){
        cudaStream_t s1;//Definicion del objeto stream para hacer la concurrencia en el kernel
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);//Encolar la tarea en s1
        cudaStreamDestroy(s1);
    }
}

void cuda_quicksort(unsigned int *data, unsigned int nitems){
    
    int left = 0;
    int right = nitems-1;
    
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));
	simple_quicksort<<< 1, 1 >>>(data, left, right, 0);
    checkCudaErrors(cudaDeviceSynchronize());
}

int main(){
	
	unsigned int * N;
	unsigned int n = SIZE;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaMallocHost(&N,n);
	
	srand(2047);
	for(int i=0;i<n;i++)N[i]=rand()%n;
	
	cudaEventRecord(start);
	cuda_quicksort(N,n);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	bool check=true;
	for(int i=0;i<n-1;i++)if(N[i]>N[i+1]){
		check=false;
		std::cout << "Error en N[" << i << "] y N[" << i+1 << "], con valores " << N[i] << " y " << N[i] << " \n";
	}
	if(check)std::cout << "Ordenamiento correcto\n";
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	std::cout << "Tiempo: " << milliseconds << " ms\n";
	cudaFreeHost(N);
	
	return 0;
}
