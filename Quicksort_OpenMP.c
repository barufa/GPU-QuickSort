#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <unistd.h>
#define N 30000000
#define M 50000

int partition(int *v, int b, int t){
    int pivote, valor_pivote,i,temp;

    pivote = b;//Inicializo el pivote
    valor_pivote = v[b];
    for(i=b+1; i<=t; i++){
        if(v[i] < valor_pivote){//Swapeo la posicion pivote con la posicion i
            pivote++;
            temp=v[i];
			v[i]=v[pivote];
			v[pivote]=temp;
        }
    }
    temp=v[b];//La finalizar coloco el pivote en su lugar
    v[b]=v[pivote];
    v[pivote]=temp;
    
    return pivote;//Devuelvo la posicion final del pivote
}

void Quicksortaux(int* v, int b, int t){
    
    if(b < t){
        int pivote=partition(v, b, t);
	    int size = (pivote-1)-b;
	    #pragma omp task if(size>=M)
	    {
			Quicksortaux(v,b,pivote-1);
		}
		Quicksortaux(v,pivote+1,t);
    }
    
    return;
}

void Quicksort(int* v, int b, int t){
		
	//~ Ordenamos dentro de parallel region
	#pragma omp parallel
	{
		//~ Como solo queremos ordenar el arreglo una vez, tenemos que agregar single para que haga la primer llamada a Quicksortaux una unica vez
		//~ Usamos nowait porque no hay necesidad de sincronizacion al final de la region
		#pragma omp single nowait
		{
			Quicksortaux(v,b,t);
		}
	}
	
	return;
}

int main(void){
    int *a,i;
    srand(time(0)^(getpid()));
    a = malloc(N*sizeof(int));
    for(i=0;i<N;i++)a[i]=rand()%N+1;
    
    double t=omp_get_wtime();//Inicializo la variable t para medir el tiempo de ejecucion
    Quicksort(a,0,N-1);//Lo ordeno
    t = omp_get_wtime()-t;//Calculo la diferencia
    printf("Tiempo : %f ms.\n",t*1000);
    
    free(a);
    
    return 0;
}
