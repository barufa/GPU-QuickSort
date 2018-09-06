#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#define SIZE 10000

__global__ void reduce(int * vector,int size,int pot){
	
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int salto = pot/2;
	
	while(salto){
		if(idx<salto && idx+salto<size){
			vector[idx]=vector[idx]+vector[idx+salto];
		} 
		__syncthreads();
		salto=salto/2;
	}
	
	return;
}

int main(void){
	
	int *N,*CU_N,len=SIZE,size=1;
	cudaEvent_t start, stop;
	
	cudaMallocHost(&N,len*sizeof(int));
	for(int i=0;i<len;i++)N[i]=1;
	while(size<len)size=(size<<1);
	
	puts("Arreglo inicializado");
	printf("Len:%d Size:%d\n",len,size);
	
	cudaMalloc(&CU_N,len*sizeof(int));
	cudaMemcpy(CU_N,N,len*sizeof(int),cudaMemcpyHostToDevice);
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	reduce<<<(len+127)/128,128>>>(CU_N,len,size);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaMemcpy(N,CU_N,sizeof(int),cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Tiempo : %f ms.\n",milliseconds);
	printf("Resultado: %d\n",N[0]);
	//~ for(int i=0;i<len;i++)printf("%d%c",N[i]," \n"[i==len-1]);
	
	cudaFreeHost(N);
	cudaFree(CU_N);
	
}


