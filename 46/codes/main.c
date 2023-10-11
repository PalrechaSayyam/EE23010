#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void uniform(char *str, int len);
void desiredDist(char *input_file, char *output_file, double *range, int length);
double desired_prob(char *req, double lower, double upper);
void shuffle(char *req, int n);

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

        if (count >= 0 && count < (int)(length * (1 - p) / 2)) {
            result = (double)(x * r[0]);
        } else if (count >= (int)(length * (1 - p) / 2) && count < (int)(length * (1 + p) / 2)) {
            result = (double)(r[1]);
        } else if (count >= (int)(length * (1 + p) / 2) && count < (int)(length)) {
            result = (double)(x);
        }

        fprintf(dp, "%lf\n", result);
        count++;
    }

    fclose(fp);
    fclose(dp);
}

double desired_prob(char *req, double lower, double upper) {
    FILE *dp;
    dp = fopen(req, "r");
    double x = 0.0;
    double prob = 0.0;

    int des_count = 0;
    int act_count = 0;

    while (fscanf(dp, "%lf", &x) != EOF) {
        if (x > lower && x < upper) {
            des_count++;
        }
        act_count++;
    }

    fclose(dp);
    prob = (double)des_count / (double)act_count;
    return prob;
}

void shuffle(char *req, int n) {
    FILE *dp = fopen(req, "r");
    double *numbers = (double *)malloc(n * sizeof(double));

    if (numbers == NULL) {
        perror("Memory allocation error");
        return;
    }

    int i = 0;
    while (fscanf(dp, "%lf", &numbers[i]) != EOF && i < n) {
        i++;
    }
    fclose(dp);

    srand(time(NULL));
    i = 0;
    int j;
    double temp;

    for (i = n - 1; i > 0; i--) {
        j = rand() % (i + 1);
        temp = numbers[i];
        numbers[i] = numbers[j];
        numbers[j] = temp;
    }
    dp = fopen(req, "w");

    for (i = 0; i < n; i++) {
        fprintf(dp, "%lf\n", numbers[i]);
    }

    fclose(dp);
    free(numbers);
}

int main(void) {
    int len = 100000;
    double range[3] = {-1.0, 0.0, 1.0};

    uniform("uni.dat", len);
    desiredDist("uni.dat", "des_dist.dat", range, len);

    shuffle("des_dist.dat", len);

    double n = 1000000.0;
    double lb = -0.5 - 1.0 / n;
    double ub = 1.0 / n;
    double act_prob = 5.0 / 8;
    double sim_prob = desired_prob("des_dist.dat", lb, ub);

    printf("The Actual Probability %lf\n", act_prob);
    printf("The Simulated Probability %lf\n", sim_prob);

    return 0;
}

