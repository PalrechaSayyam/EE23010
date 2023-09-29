#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void uniform(char *str, int len);
double mean(char *str);

void uniform(char *str, int len){
	int i;
	FILE *fp;
	fp = fopen(str,"w");
	
	for (i = 0; i < len; i++){
		//Generating the given distribution from the uniform distribution(0,1)
		fprintf(fp,"%lf\n",-sqrt(3)+2*sqrt(3)*(double)rand()/RAND_MAX);
	}
	
	fclose(fp);
}

double mean(char *str){
	int i = 0,c;
	FILE *fp;
	double x, temp = 0.0;
	fp = fopen(str,"r");
	
	while(fscanf(fp,"%lf",&x)!=EOF){
		i=i+1;
		temp = temp+x;
	}
	
	fclose(fp);
	temp = temp/(i-1);
	return temp;
}

double var(char *str){
	int i = 0,c;
	FILE *fp;
	double x, temp = 0.0;
	double exp = mean(str);
	fp = fopen(str,"r");
	
	while(fscanf(fp,"%lf",&x)!=EOF){
		i=i+1;
		double sqr = x - exp;
		temp += sqr * sqr;	
	}
	
	fclose(fp);
	temp = temp/(i-1);
	return temp;
}

int main(void){

	uniform("uni.dat", 1000000);
	
	//Variance of the given uniform distribution
	printf("%lf\n",var("uni.dat"));
	
	return 0;
	
}

