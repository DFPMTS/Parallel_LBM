#include "calc.h"

/* set inlets velocity there are two type inlets*/
int set_inlets(const t_param params, float *inlets) {
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
float av_velocity(const t_param params, t_speed *cells, int *obstacles) {
  int tot_cells = 0; /* no. of cells used in calculation */
  float tot_u;       /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

/* loop over all non-blocked cells */
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

  return tot_u / (float)tot_cells;
}

/* calculate reynold number */
float calc_reynolds(const t_param params, t_speed *cells, int *obstacles) {
  return av_velocity(params, cells, obstacles) * (float)(params.ny) /
         params.viscosity;
}