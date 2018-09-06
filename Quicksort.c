#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <unistd.h>

#define N 1000000000

int Quicksortaux(int *v, int b, int t){
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

void Quicksort(int* v, int b, int t){
    if(b < t){
        int pivote=Quicksortaux(v, b, t);//Ordeno parcialmente el arreglo
	    Quicksort(v,b,pivote-1);//Me llamo recursivamente sobre
        Quicksort(v,pivote+1,t);//las mitades restantes
    }
}

int main(int argc, char *argv[]){
	
	if(argc!=2)return 0;
	const int MAXN = atoi(argv[1]);
	srand(time(0)^(getpid()));
	 
    int *a,i;
    a = malloc(MAXN*sizeof(int));
    for(i=0;i<MAXN;i++)a[i]=rand()%MAXN+1;
    
    double t=omp_get_wtime();//Inicializo la variable t para medir el tiempo de ejecucion
    Quicksort(a,0,MAXN-1);//Lo ordeno
    t = omp_get_wtime()-t;//Calculo la diferencia
    printf("Tiempo : %f ms.\n",t*1000);
    for(i=0;i<MAXN-1;i++)if(a[i]>a[i+1])puts("Error");
    free(a);
    
    return 0;
}
