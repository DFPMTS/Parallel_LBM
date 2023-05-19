#include "d2q9_bgk.h"
#include "immintrin.h"
#include "types.h"
#include <omp.h>
#include <xmmintrin.h>

/* The main processes in one step */
int collision(const t_param params, t_speed *cells, t_speed *tmp_cells,
              int *obstacles);
int streaming(const t_param params, t_speed *cells, t_speed *tmp_cells);
int obstacle(const t_param params, t_speed *cells, t_speed *tmp_cells,
             int *obstacles);
int boundary(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** collision(), obstacle(), streaming() & boundary()
*/
int timestep(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets, int *obstacles) {
  /* The main time overhead, you should mainly optimize these processes. */
  collision(params, cells, tmp_cells, obstacles);
  // obstacle(params, cells, tmp_cells, obstacles);
  streaming(params, cells, tmp_cells);
  boundary(params, cells, tmp_cells, inlets);
  return EXIT_SUCCESS;
}

/*
** The collision of fluids in the cell is calculated using
** the local equilibrium distribution and relaxation process
*/

int collision(const t_param params, t_speed *cells, t_speed *tmp_cells,
              int *obstacles) {
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */

  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */

  __m256 _1 = _mm256_set1_ps(1.f);
  __m256 c = _mm256_set1_ps(c_sq);
  __m256 _2_c_c = _mm256_set1_ps(2.f * c_sq * c_sq);
  __m256 w = _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2);
  __m256 omega = _mm256_set1_ps(params.omega);

#pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++) {
    float buffer[8];
    for (int ii = 0; ii < params.nx; ii++) {
      if (!obstacles[ii + jj * params.nx]) {
        /* compute local density total */
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

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* equilibrium densities */
        float d_equ;
        /* zero velocity density: weight w0 */

        d_equ = w0 * local_density * (1.f - u_sq / (2.f * c_sq));

        __m256 x = _mm256_setr_ps(u_x, u_y, -u_x, -u_y, u_x + u_y, -u_x + u_y,
                                  -u_x - u_y, u_x - u_y);

        __m256 res = _mm256_add_ps(
            _mm256_add_ps(_1, _mm256_div_ps(x, c)),
            _mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(x, x), _2_c_c),
                          _mm256_set1_ps(u_sq / (2.f * c_sq))));
        res =
            _mm256_mul_ps(_mm256_mul_ps(res, _mm256_set1_ps(local_density)), w);
        /* relaxation step */
        tmp_cells->speeds[0][ii + jj * params.nx] =
            cells->speeds[0][ii + jj * params.nx] +
            params.omega * (d_equ - cells->speeds[0][ii + jj * params.nx]);
        __m256 c_s = _mm256_setr_ps(cells->speeds[1][ii + jj * params.nx],
                                    cells->speeds[2][ii + jj * params.nx],
                                    cells->speeds[3][ii + jj * params.nx],
                                    cells->speeds[4][ii + jj * params.nx],
                                    cells->speeds[5][ii + jj * params.nx],
                                    cells->speeds[6][ii + jj * params.nx],
                                    cells->speeds[7][ii + jj * params.nx],
                                    cells->speeds[8][ii + jj * params.nx]);
        // __m256 c_s = _mm256_loadu_ps(cells[ii + jj * params.nx].speeds +
        // 1);
        res = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(res, c_s), omega), c_s);
        _mm256_storeu_ps(buffer, res);
        for (int k = 1; k < NSPEEDS; ++k)
          tmp_cells->speeds[k][ii + jj * params.nx] = buffer[k - 1];
      } else {
        tmp_cells->speeds[0][ii + jj * params.nx] =
            cells->speeds[0][ii + jj * params.nx];
        tmp_cells->speeds[1][ii + jj * params.nx] =
            cells->speeds[3][ii + jj * params.nx];
        tmp_cells->speeds[3][ii + jj * params.nx] =
            cells->speeds[1][ii + jj * params.nx];
        tmp_cells->speeds[2][ii + jj * params.nx] =
            cells->speeds[4][ii + jj * params.nx];
        tmp_cells->speeds[4][ii + jj * params.nx] =
            cells->speeds[2][ii + jj * params.nx];
        tmp_cells->speeds[5][ii + jj * params.nx] =
            cells->speeds[7][ii + jj * params.nx];
        tmp_cells->speeds[7][ii + jj * params.nx] =
            cells->speeds[5][ii + jj * params.nx];
        tmp_cells->speeds[6][ii + jj * params.nx] =
            cells->speeds[8][ii + jj * params.nx];
        tmp_cells->speeds[8][ii + jj * params.nx] =
            cells->speeds[6][ii + jj * params.nx];
      }
    }
  }
  return EXIT_SUCCESS;
}

/*
** For obstacles, mirror their speed.
*/
int obstacle(const t_param params, t_speed *cells, t_speed *tmp_cells,
             int *obstacles) {

/* loop over the cells in the grid */
#pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* if the cell contains an obstacle */
      if (obstacles[jj * params.nx + ii]) {
        /* called after collision, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells->speeds[0][ii + jj * params.nx] =
            cells->speeds[0][ii + jj * params.nx];
        tmp_cells->speeds[1][ii + jj * params.nx] =
            cells->speeds[3][ii + jj * params.nx];
        tmp_cells->speeds[3][ii + jj * params.nx] =
            cells->speeds[1][ii + jj * params.nx];
        tmp_cells->speeds[2][ii + jj * params.nx] =
            cells->speeds[4][ii + jj * params.nx];
        tmp_cells->speeds[4][ii + jj * params.nx] =
            cells->speeds[2][ii + jj * params.nx];
        tmp_cells->speeds[5][ii + jj * params.nx] =
            cells->speeds[7][ii + jj * params.nx];
        tmp_cells->speeds[7][ii + jj * params.nx] =
            cells->speeds[5][ii + jj * params.nx];
        tmp_cells->speeds[6][ii + jj * params.nx] =
            cells->speeds[8][ii + jj * params.nx];
        tmp_cells->speeds[8][ii + jj * params.nx] =
            cells->speeds[6][ii + jj * params.nx];
      }
    }
  }
  return EXIT_SUCCESS;
}

/*
** Particles flow to the corresponding cell according to their speed direaction.
*/
int streaming(const t_param params, t_speed *cells, t_speed *tmp_cells) {
/* loop over _all_ cells */
#pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      cells->speeds[0][ii + jj * params.nx] =
          tmp_cells
              ->speeds[0][ii + jj * params.nx]; /* central cell, no movement */
      cells->speeds[1][x_e + jj * params.nx] =
          tmp_cells->speeds[1][ii + jj * params.nx]; /* east */
      cells->speeds[2][ii + y_n * params.nx] =
          tmp_cells->speeds[2][ii + jj * params.nx]; /* north */
      cells->speeds[3][x_w + jj * params.nx] =
          tmp_cells->speeds[3][ii + jj * params.nx]; /* west */
      cells->speeds[4][ii + y_s * params.nx] =
          tmp_cells->speeds[4][ii + jj * params.nx]; /* south */
      cells->speeds[5][x_e + y_n * params.nx] =
          tmp_cells->speeds[5][ii + jj * params.nx]; /* north-east */
      cells->speeds[6][x_w + y_n * params.nx] =
          tmp_cells->speeds[6][ii + jj * params.nx]; /* north-west */
      cells->speeds[7][x_w + y_s * params.nx] =
          tmp_cells->speeds[7][ii + jj * params.nx]; /* south-west */
      cells->speeds[8][x_e + y_s * params.nx] =
          tmp_cells->speeds[8][ii + jj * params.nx]; /* south-east */
    }
  }

  return EXIT_SUCCESS;
}

/*
** Work with boundary conditions. The upper and lower boundaries use the rebound
*plane,
** the left border is the inlet of fixed speed, and
** the right border is the open outlet of the first-order approximation.
*/
int boundary(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets) {
  /* Set the constant coefficient */
  const float cst1 = 2.0 / 3.0;
  const float cst2 = 1.0 / 6.0;
  const float cst3 = 1.0 / 2.0;

  int ii, jj;
  float local_density;

  // top wall (bounce)
  jj = params.ny - 1;
#pragma omp parallel for
  for (ii = 0; ii < params.nx; ii++) {
    cells->speeds[4][ii + jj * params.nx] =
        tmp_cells->speeds[2][ii + jj * params.nx];
    cells->speeds[7][ii + jj * params.nx] =
        tmp_cells->speeds[5][ii + jj * params.nx];
    cells->speeds[8][ii + jj * params.nx] =
        tmp_cells->speeds[6][ii + jj * params.nx];
  }

  // bottom wall (bounce)
  jj = 0;
#pragma omp parallel for
  for (ii = 0; ii < params.nx; ii++) {
    cells->speeds[2][ii + jj * params.nx] =
        tmp_cells->speeds[4][ii + jj * params.nx];
    cells->speeds[5][ii + jj * params.nx] =
        tmp_cells->speeds[7][ii + jj * params.nx];
    cells->speeds[6][ii + jj * params.nx] =
        tmp_cells->speeds[8][ii + jj * params.nx];
  }

  // left wall (inlet)
  ii = 0;
#pragma omp parallel for
  for (jj = 0; jj < params.ny; jj++) {
    local_density = (cells->speeds[0][ii + jj * params.nx] +
                     cells->speeds[2][ii + jj * params.nx] +
                     cells->speeds[4][ii + jj * params.nx] +
                     2.0 * cells->speeds[3][ii + jj * params.nx] +
                     2.0 * cells->speeds[6][ii + jj * params.nx] +
                     2.0 * cells->speeds[7][ii + jj * params.nx]) /
                    (1.0 - inlets[jj]);

    cells->speeds[1][ii + jj * params.nx] =
        cells->speeds[3][ii + jj * params.nx] +
        cst1 * local_density * inlets[jj];

    cells->speeds[5][ii + jj * params.nx] =
        cells->speeds[7][ii + jj * params.nx] -
        cst3 * (cells->speeds[2][ii + jj * params.nx] -
                cells->speeds[4][ii + jj * params.nx]) +
        cst2 * local_density * inlets[jj];

    cells->speeds[8][ii + jj * params.nx] =
        cells->speeds[6][ii + jj * params.nx] +
        cst3 * (cells->speeds[2][ii + jj * params.nx] -
                cells->speeds[4][ii + jj * params.nx]) +
        cst2 * local_density * inlets[jj];
  }

  // right wall (outlet)
  ii = params.nx - 1;
#pragma omp parallel for
  for (jj = 0; jj < params.ny; jj++) {
    for (int kk = 0; kk < NSPEEDS; kk++) {
      cells->speeds[kk][ii + jj * params.nx] =
          cells->speeds[kk][ii - 1 + jj * params.nx];
    }
  }

  return EXIT_SUCCESS;
}
