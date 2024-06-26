#ifndef TYPES_H
#define TYPES_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>

#define NSPEEDS 9
#define NUM_THREADS 28

typedef struct {
  int nx;          /* no. of cells in x-direction */
  int ny;          /* no. of cells in y-direction */
  int maxIters;    /* no. of iterations */
  float density;   /* density per cell */
  float viscosity; /* kinematic viscosity of fluid */
  float velocity;  /* inlet velocity */
  int type;        /* inlet type */
  float omega;     /* relaxation parameter */
} t_param;

/* struct to hold the distribution of different speeds */
typedef struct {
  float *speeds[NSPEEDS];
} t_speed;

typedef struct {
  float speeds[NSPEEDS];
} t_speed_aos;

#endif