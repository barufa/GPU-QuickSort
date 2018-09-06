#include <stdio.h>
#define N 1000

//Se define el kernel(funcion) add, que duplica los valores de a almacenandolos en b
__global__
void add(int *a, int *b) {
    int i = blockIdx.x;//Tenemos un solo hilo por bloque, por lo que el numero de bloque es suficiente para distinguir el proceso
    if (i<N) {//Nos seguramos de no salir del arreglo
        b[i] = 2*a[i];
    }
}

int main() {
    //Se creean los arreglos en el CPU
    int * ha = (int *)malloc(N*sizeof(int));
	
	for (int i = 0; i<N; ++i) {
        ha[i] = i;
    }
	
	//Se crean los arreglos en la GPU
    int *da, *db;
    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));
	
	//Se hace una copia de los datos, pasandoselos a la GPU
    cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);

    //Se hace la llamada a la funcion add(kernel) desde el CPU, con N bloques con un hilo cada uno
    add<<<N, 1>>>(da, db);
	
	//Se traen los resultados desde la GPU al CPU para serguir con el programa
    cudaMemcpy(ha, db, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; ++i) {
        printf("%d\n", ha[i]);
    }

    cudaFree(da);
    cudaFree(db);
	free(ha);
    return 0;
}

