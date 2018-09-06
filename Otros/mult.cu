#include <stdio.h>
#include <stdlib.h>

__global__ void multiplication(int * A,int * B,int * C,int N){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	
	if(row<N && col<N){//Si no me fui del arreglo
		 int sum=0;
		 for(int i=0;i<N;i++){
			 sum+=A[row*N+i]*B[i*N+col];
		 }
		 C[row*N+col]=sum;
	}
}

#define N 30000

int main() {
    //Se creean los arreglos en el CPU
	int * A = (int *)malloc(sizeof(int)*(N*N));
	int * B = (int *)malloc(sizeof(int)*(N*N));
	int * C = (int *)malloc(sizeof(int)*(N*N));
	
	//~ srand(333);
	//~ for(int i=0;i<N;i++)for(int j=0;j<N;j++)A[i*N + j]=rand()%100;
	//~ for(int i=0;i<N;i++)for(int j=0;j<N;j++)B[i*N + j]=rand()%100;
	//~ for(int i=0;i<N;i++)for(int j=0;j<N;j++)C[i*N + j]=rand()%100;


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	
	//Se crean los arreglos en la GPU
	int *CU_A,*CU_B,*CU_C;
    cudaMalloc((void **)&CU_A,sizeof(int)*(N*N));
    cudaMalloc((void **)&CU_B,sizeof(int)*(N*N));
    cudaMalloc((void **)&CU_C,sizeof(int)*(N*N));
	cudaMemcpy(CU_A,A, N*N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(CU_B,B, N*N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(CU_C,C, N*N*sizeof(int), cudaMemcpyHostToDevice);
	
	//Se ejecuta la funcion y se mide su performance
	multiplication<<<(N+255)/512,256>>>(CU_A,CU_B,CU_C,N);
	cudaEventRecord(stop);
	
	//Se traen los resultados
	cudaMemcpy(C,CU_C,N*sizeof(int),cudaMemcpyDeviceToHost);
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	printf("Tiempo para matrices de %dx%d : %f ms.\n",N,N,milliseconds);
	
	//Validacion de los resultados
	//~ int t=1;
	//~ for(int i=0;i<N;i++)for(int j=0;j<N;j++){
		//~ int sum = 0;
		//~ for(int k=0;k<N;k++)sum+=A[i*N+k]*B[k*N+j];
		//~ if(C[i*N+j]!=sum)t=0;
	//~ }
	//~ if(t)puts("Algo salio mal");
	//~ else puts("YEAH!");
	
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

