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
		//Generating the standard uniform distribution
		fprintf(fp,"%lf\n",(double)rand()/RAND_MAX);
	}
	
	fclose(fp);
}

double desiredDist(char *str, char *req, double *r){
	FILE *fp;
	FILE *dp;
	fp = fopen(str,"r");
	dp = fopen(req,"w");
	double x = 0.0;
	
	while(fscanf(fp,"%lf",&x)!=EOF){
		//generating desired distribution
		fprintf(dp,"%lf\n",(r[1]-r[0])*x + r[0]);
	}
	fclose(fp);
	fclose(dp);
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
	
	double range[2] = {-sqrt(3),sqrt(3)};

	uniform("uni.dat", 1000000);
	desiredDist("uni.dat","des_dist.dat",range);
	
	//Variance of the desired distribution
	printf("%lf\n",var("des_dist.dat"));
	
	return 0;
	
}

