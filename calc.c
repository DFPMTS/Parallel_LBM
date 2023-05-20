#include "calc.h"

/* set inlets velocity there are two type inlets*/
int set_inlets(const t_param params, float *inlets) {
#pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++) {
    if (!params.type)
      inlets[jj] = params.velocity; // homogeneous
    else
      inlets[jj] = params.velocity * 4.0 *
                   ((1 - ((float)jj) / params.ny) * ((float)(jj + 1)) /
                    params.ny); // parabolic
  }
  return EXIT_SUCCESS;
}

/* compute average velocity of whole grid, ignore grids with obstacles. */
float av_velocity(const t_param params, t_speed *cells, int *obstacles,
                  t_speed_aos *cells_aos, int type) {
  int tot_cells = 0; /* no. of cells used in calculation */
  float tot_u;       /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  if (type == 0) {
#pragma omp parallel for
    for (int jj = 0; jj < params.ny; jj++) {
      for (int ii = 0; ii < params.nx; ii++) {
        /* ignore occupied cells */
        if (!obstacles[ii + jj * params.nx]) {
          float local_density = 0.f;

          for (int kk = 0; kk < NSPEEDS; kk++) {
            local_density += cells->speeds[kk][ii + jj * params.nx];
          }

          /* compute x velocity component */
          float u_x = (cells->speeds[1][ii + jj * params.nx] +
                       cells->speeds[5][ii + jj * params.nx] +
                       cells->speeds[8][ii + jj * params.nx] -
                       (cells->speeds[3][ii + jj * params.nx] +
                        cells->speeds[6][ii + jj * params.nx] +
                        cells->speeds[7][ii + jj * params.nx])) /
                      local_density;
          /* compute y velocity component */
          float u_y = (cells->speeds[2][ii + jj * params.nx] +
                       cells->speeds[5][ii + jj * params.nx] +
                       cells->speeds[6][ii + jj * params.nx] -
                       (cells->speeds[4][ii + jj * params.nx] +
                        cells->speeds[7][ii + jj * params.nx] +
                        cells->speeds[8][ii + jj * params.nx])) /
                      local_density;
          /* accumulate the norm of x- and y- velocity components */
          tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
          /* increase counter of inspected cells */
          ++tot_cells;
        }
      }
    }
  } else if (type == 1) {
#pragma omp parallel for
    for (int jj = 0; jj < params.ny; jj++) {
      for (int ii = 0; ii < params.nx; ii++) {
        /* ignore occupied cells */
        if (!obstacles[ii + jj * params.nx]) {
          /* local density total */
          float local_density = 0.f;

          for (int kk = 0; kk < NSPEEDS; kk++) {
            local_density += cells_aos[ii + jj * params.nx].speeds[kk];
          }

          /* x-component of velocity */
          float u_x = (cells_aos[ii + jj * params.nx].speeds[1] +
                       cells_aos[ii + jj * params.nx].speeds[5] +
                       cells_aos[ii + jj * params.nx].speeds[8] -
                       (cells_aos[ii + jj * params.nx].speeds[3] +
                        cells_aos[ii + jj * params.nx].speeds[6] +
                        cells_aos[ii + jj * params.nx].speeds[7])) /
                      local_density;
          /* compute y velocity component */
          float u_y = (cells_aos[ii + jj * params.nx].speeds[2] +
                       cells_aos[ii + jj * params.nx].speeds[5] +
                       cells_aos[ii + jj * params.nx].speeds[6] -
                       (cells_aos[ii + jj * params.nx].speeds[4] +
                        cells_aos[ii + jj * params.nx].speeds[7] +
                        cells_aos[ii + jj * params.nx].speeds[8])) /
                      local_density;
          /* accumulate the norm of x- and y- velocity components */
          tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
          /* increase counter of inspected cells */
          ++tot_cells;
        }
      }
    }
  }

  return tot_u / (float)tot_cells;
}

/* calculate reynold number */
float calc_reynolds(const t_param params, t_speed *cells, int *obstacles,
                    t_speed_aos *cells_aos, int type) {
  return av_velocity(params, cells, obstacles, cells_aos, type) *
         (float)(params.ny) / params.viscosity;
}