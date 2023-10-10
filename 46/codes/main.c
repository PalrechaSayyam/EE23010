#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void uniform(char *str, int len);
void desiredDist(char *input_file, char *output_file, double *range, int length);
double desired_prob(char *req, double lower, double upper);

void uniform(char *str, int len) {
    int i;
    FILE *fp;
    fp = fopen(str, "w");

    for (i = 0; i < len; i++) {
        // Generating the standard uniform distribution
        fprintf(fp, "%lf\n", (double)rand() / RAND_MAX);
    }

    fclose(fp);
}

void desiredDist(char *str, char *req, double *r, int length) {
    FILE *fp, *dp;
    double x = 0.0;
    double result = 0.0;
    double p = 0.5;
    
    fp = fopen(str, "r");
    dp = fopen(req, "w");
    
    int count = 0;

    while (fscanf(fp, "%lf", &x) != EOF) {
    
    	if(count>=0.0 && count<(int)(length*(1-p)/2)){
    		result = (double)(x+r[0]);
    	}else if(count>= (int)(length*(1-p)/2) && count<(int)(length*(1+p)/2)){
		result = (double)(r[1]);
	}else if(count>(int)(length*(1+p)/2) && count<(int)(length)){
		result = (double)(x);
	}
	
	fprintf(dp, "%lf\n", result);
	count++;
	
    }

    fclose(fp);
    fclose(dp);
}

double desired_prob(char *req, double lower, double upper){
	FILE *dp;
	dp = fopen(req, "r");
	double x = 0.0;
	double prob = 0.0;
	
	int des_count = 0;
	int act_count = 0;
	
	while (fscanf(dp, "%lf", &x) != EOF){
		if(x>lower && x < upper){
			des_count++;
		}
		act_count++;
	}
	
	fclose(dp);
	prob = (double)des_count/(double)act_count;
	return prob;
}

int main(void) {
    int len = 100000;
    double range[3] = {-1.0, 0.0, 1.0};

    uniform("uni.dat", len);
    desiredDist("uni.dat", "des_dist.dat", range, len);
    
    double n = 1000000.0;
    double lb = (double)(-1/2 - 1/n);
    double ub = (double)(1/n);
    double act_prob = (double)5/8;
    double sim_prob = desired_prob("des_dist.dat",lb,ub);
    
    printf("The Actual Probability %lf\n", act_prob);
    printf("The Simulated Probability %lf\n", sim_prob);
    
    return 0;
}

