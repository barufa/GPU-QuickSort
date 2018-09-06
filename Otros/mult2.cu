#include <stdio.h>
#include <stdlib.h>

__global__ void multiplication(int * A,int * B,int * C,int N,int M,int K){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	
	if(row<N && col<K){//Si no me fui del arreglo
		 int sum=0;
		 for(int i=0;i<M;i++){
			 sum+=A[row*N+i]*B[i*M+col];
		 }
		 C[row*N+col]=sum;
	}
}

#define N 333
#define M 444
#define K 222

int main() {
    //Se creean los arreglos en el CPU
	int * A = (int *)malloc(sizeof(int)*(N*M+1));
	int * B = (int *)malloc(sizeof(int)*(M*K+1));
	int * C = (int *)malloc(sizeof(int)*(N*K+1));
	
	srand(333);
	for(int i=0;i<N;i++)for(int j=0;j<M;j++)A[i*N + j]=rand()%100;
	for(int i=0;i<M;i++)for(int j=0;j<K;j++)B[i*M + j]=rand()%100;
	for(int i=0;i<N;i++)for(int j=0;j<K;j++)C[i*N + j]=rand()%100;
	
	//Se crean los arreglos en la GPU
	int *CU_A,*CU_B,*CU_C;
    cudaMalloc((void **)&CU_A,sizeof(int)*(N*M));
    cudaMalloc((void **)&CU_B,sizeof(int)*(M*K));
    cudaMalloc((void **)&CU_C,sizeof(int)*(N*K));
	cudaMemcpy(CU_A,A, N*N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(CU_B,B, N*N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(CU_C,C, N*N*sizeof(int), cudaMemcpyHostToDevice);
	
	//Se ejecuta la funcion y se mide su performance
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	multiplication<<<(N+255)/256,256>>>(CU_A,CU_B,CU_C,N,M,K);
	cudaEventRecord(stop);
	
	//Se traen los resultados
	cudaMemcpy(C,CU_C,N*sizeof(int),cudaMemcpyDeviceToHost);
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	printf("Tiempo para matrices de %dx%d : %f ms.\n",N,N,milliseconds);
	
	//Validacion de los resultados
	int t=1;
	for(int i=0;i<N;i++)for(int j=0;j<K;j++){
		int sum = 0;
		for(int k=0;k<M;k++)sum+=A[i*N+k]*B[k*M+j];
		if(C[i*N+j]!=sum)t=0;
	}
	if(t)puts("Algo salio mal");
	else puts("YEAH!");
	
	//Se libera la memoria
	free(A);
	free(B);
	free(C);
	cudaFree(CU_A);
	cudaFree(CU_B);
	cudaFree(CU_C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    return 0;
}

