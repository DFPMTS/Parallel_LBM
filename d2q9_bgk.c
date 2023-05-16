#include "d2q9_bgk.h"
#include "immintrin.h"
#include <omp.h>
#include <string.h>
#include <xmmintrin.h>

/* The main processes in one step */
int collision(int start_col, int end_col, const t_param params, t_speed *cells,
              t_speed *tmp_cells, int *obstacles);
int streaming(int start_col, int end_col, const t_param params, t_speed *cells,
              t_speed *tmp_cells);
int obstacle(const t_param params, t_speed *cells, t_speed *tmp_cells,
             int *obstacles);
int boundary(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets);
inline int fuse(int start_col, int end_col, const t_param params,
                t_speed *cells, t_speed *tmp_cells, int *obstacles);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** collision(), obstacle(), streaming() & boundary()
*/
int timestep(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets, int *obstacles) {
  /* The main time overhead, you should mainly optimize these processes. */

  collision(0, params.nx, params, cells, tmp_cells, obstacles);
  streaming(0, params.nx, params, cells, tmp_cells);

  boundary(params, cells, tmp_cells, inlets);

  return EXIT_SUCCESS;
}

/*
** The collision of fluids in the cell is calculated using
** the local equilibrium distribution and relaxation process
*/

#define chunk_x 64
#define chunk_y 64
inline int fuse(int start_col, int end_col, const t_param params,
                t_speed *cells, t_speed *tmp_cells, int *obstacles) {
  static const float c_sq = 1.f / 3.f; /* square of speed of sound */
  static const float w0 = 4.f / 9.f;   /* weighting factor */
  static const float w1 = 1.f / 9.f;   /* weighting factor */
  static const float w2 = 1.f / 36.f;  /* weighting factor */

  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */
  __m256 _1 = _mm256_set1_ps(1.f);
  __m256 c = _mm256_set1_ps(c_sq);
  __m256 _2_c_c = _mm256_set1_ps(2.f * c_sq * c_sq);
  __m256 w = _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2);
  __m256 omega = _mm256_set1_ps(params.omega);

#pragma omp parallel
  {
    t_speed buffer;
    int id = omp_get_thread_num();
    int col_per_thread = params.nx / omp_get_num_threads() + 1;
    int start_col = id * col_per_thread, end_col = (id + 1) * col_per_thread;
    int jj_start, jj_end, ii_start, ii_end;

    if (end_col > params.nx)
      end_col = params.nx;
    for (int j = 0; j < params.ny; j += chunk_y) {
      jj_start = j;
      jj_end = ((j + chunk_y) > params.ny) ? params.ny : (j + chunk_y);
      for (int i = start_col; i < end_col; i += chunk_x) {
        ii_start = i;
        ii_end = ((i + chunk_x) > end_col) ? end_col : (i + chunk_x);
        for (int jj = jj_start, jj_offset = 0; jj < jj_end; jj++, jj_offset++) {
          for (int ii = ii_start, ii_offset = 0; ii < ii_end;
               ii++, ii_offset++) {

            if (!obstacles[ii + jj * params.nx]) {
              /* compute local density total */
              float local_density = 0.f;

              for (int kk = 0; kk < NSPEEDS; kk++) {
                local_density += cells[ii + jj * params.nx].speeds[kk];
              }

              /* compute x velocity component */
              float u_x = (cells[ii + jj * params.nx].speeds[1] +
                           cells[ii + jj * params.nx].speeds[5] +
                           cells[ii + jj * params.nx].speeds[8] -
                           (cells[ii + jj * params.nx].speeds[3] +
                            cells[ii + jj * params.nx].speeds[6] +
                            cells[ii + jj * params.nx].speeds[7])) /
                          local_density;
              /* compute y velocity component */
              float u_y = (cells[ii + jj * params.nx].speeds[2] +
                           cells[ii + jj * params.nx].speeds[5] +
                           cells[ii + jj * params.nx].speeds[6] -
                           (cells[ii + jj * params.nx].speeds[4] +
                            cells[ii + jj * params.nx].speeds[7] +
                            cells[ii + jj * params.nx].speeds[8])) /
                          local_density;

              /* velocity squared */
              float u_sq = u_x * u_x + u_y * u_y;

              /* equilibrium densities */
              float d_equ;
              /* zero velocity density: weight w0 */

              d_equ = w0 * local_density * (1.f - u_sq / (2.f * c_sq));

              __m256 x = _mm256_setr_ps(u_x, u_y, -u_x, -u_y, u_x + u_y,
                                        -u_x + u_y, -u_x - u_y, u_x - u_y);

              __m256 res = _mm256_add_ps(
                  _mm256_add_ps(_1, _mm256_div_ps(x, c)),
                  _mm256_sub_ps(
                      _mm256_div_ps(_mm256_mul_ps(x, x),
                                    _mm256_set1_ps(2.f * c_sq * c_sq)),
                      _mm256_set1_ps(u_sq / (2.f * c_sq))));
              res = _mm256_mul_ps(
                  _mm256_mul_ps(res, _mm256_set1_ps(local_density)),
                  _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2));
              /* relaxation step */
              tmp_cells[ii + jj * params.nx].speeds[0] =
                  cells[ii + jj * params.nx].speeds[0] +
                  params.omega * (d_equ - cells[ii + jj * params.nx].speeds[0]);
              __m256 c_s =
                  _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);
              res = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(res, c_s), omega),
                                  c_s);
              _mm256_storeu_ps(tmp_cells[ii + jj * params.nx].speeds + 1, res);
            } else {
              tmp_cells[ii + jj * params.nx].speeds[0] =
                  cells[ii + jj * params.nx].speeds[0];
              tmp_cells[ii + jj * params.nx].speeds[1] =
                  cells[ii + jj * params.nx].speeds[3];
              tmp_cells[ii + jj * params.nx].speeds[2] =
                  cells[ii + jj * params.nx].speeds[4];
              tmp_cells[ii + jj * params.nx].speeds[3] =
                  cells[ii + jj * params.nx].speeds[1];
              tmp_cells[ii + jj * params.nx].speeds[4] =
                  cells[ii + jj * params.nx].speeds[2];
              tmp_cells[ii + jj * params.nx].speeds[5] =
                  cells[ii + jj * params.nx].speeds[7];
              tmp_cells[ii + jj * params.nx].speeds[6] =
                  cells[ii + jj * params.nx].speeds[8];
              tmp_cells[ii + jj * params.nx].speeds[7] =
                  cells[ii + jj * params.nx].speeds[5];
              tmp_cells[ii + jj * params.nx].speeds[8] =
                  cells[ii + jj * params.nx].speeds[6];
            }
          }
        }
      }
    }
  }

  return EXIT_SUCCESS;
}

int collision(int start_col, int end_col, const t_param params, t_speed *cells,
              t_speed *tmp_cells, int *obstacles) {
  static const float c_sq = 1.f / 3.f; /* square of speed of sound */
  static const float w0 = 4.f / 9.f;   /* weighting factor */
  static const float w1 = 1.f / 9.f;   /* weighting factor */
  static const float w2 = 1.f / 36.f;  /* weighting factor */

  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */
  __m256 _1 = _mm256_set1_ps(1.f);
  __m256 c = _mm256_set1_ps(c_sq);
  __m256 _2_c_c = _mm256_set1_ps(2.f * c_sq * c_sq);
  __m256 w = _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2);
  __m256 omega = _mm256_set1_ps(params.omega);

#pragma omp parallel
  {
    t_speed buffer;
    int id = omp_get_thread_num();
    int col_per_thread = params.nx / omp_get_num_threads() + 1;
    int start_col = id * col_per_thread, end_col = (id + 1) * col_per_thread;
    int jj_start, jj_end, ii_start, ii_end;

    if (end_col > params.nx)
      end_col = params.nx;
    for (int j = 0; j < params.ny; j += chunk_y) {
      jj_start = j;
      jj_end = ((j + chunk_y) > params.ny) ? params.ny : (j + chunk_y);
      for (int i = start_col; i < end_col; i += chunk_x) {
        ii_start = i;
        ii_end = ((i + chunk_x) > end_col) ? end_col : (i + chunk_x);
        for (int jj = jj_start, jj_offset = 0; jj < jj_end; jj++, jj_offset++) {
          for (int ii = ii_start, ii_offset = 0; ii < ii_end;
               ii++, ii_offset++) {

            if (!obstacles[ii + jj * params.nx]) {
              /* compute local density total */
              float local_density = 0.f;

              for (int kk = 0; kk < NSPEEDS; kk++) {
                local_density += cells[ii + jj * params.nx].speeds[kk];
              }

              /* compute x velocity component */
              float u_x = (cells[ii + jj * params.nx].speeds[1] +
                           cells[ii + jj * params.nx].speeds[5] +
                           cells[ii + jj * params.nx].speeds[8] -
                           (cells[ii + jj * params.nx].speeds[3] +
                            cells[ii + jj * params.nx].speeds[6] +
                            cells[ii + jj * params.nx].speeds[7])) /
                          local_density;
              /* compute y velocity component */
              float u_y = (cells[ii + jj * params.nx].speeds[2] +
                           cells[ii + jj * params.nx].speeds[5] +
                           cells[ii + jj * params.nx].speeds[6] -
                           (cells[ii + jj * params.nx].speeds[4] +
                            cells[ii + jj * params.nx].speeds[7] +
                            cells[ii + jj * params.nx].speeds[8])) /
                          local_density;

              /* velocity squared */
              float u_sq = u_x * u_x + u_y * u_y;

              /* equilibrium densities */
              float d_equ;
              /* zero velocity density: weight w0 */

              d_equ = w0 * local_density * (1.f - u_sq / (2.f * c_sq));

              __m256 x = _mm256_setr_ps(u_x, u_y, -u_x, -u_y, u_x + u_y,
                                        -u_x + u_y, -u_x - u_y, u_x - u_y);

              __m256 res = _mm256_add_ps(
                  _mm256_add_ps(_1, _mm256_div_ps(x, c)),
                  _mm256_sub_ps(
                      _mm256_div_ps(_mm256_mul_ps(x, x),
                                    _mm256_set1_ps(2.f * c_sq * c_sq)),
                      _mm256_set1_ps(u_sq / (2.f * c_sq))));
              res = _mm256_mul_ps(
                  _mm256_mul_ps(res, _mm256_set1_ps(local_density)),
                  _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2));
              /* relaxation step */
              tmp_cells[ii + jj * params.nx].speeds[0] =
                  cells[ii + jj * params.nx].speeds[0] +
                  params.omega * (d_equ - cells[ii + jj * params.nx].speeds[0]);
              __m256 c_s =
                  _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);
              res = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(res, c_s), omega),
                                  c_s);
              _mm256_storeu_ps(tmp_cells[ii + jj * params.nx].speeds + 1, res);
            } else {
              tmp_cells[ii + jj * params.nx].speeds[0] =
                  cells[ii + jj * params.nx].speeds[0];
              tmp_cells[ii + jj * params.nx].speeds[1] =
                  cells[ii + jj * params.nx].speeds[3];
              tmp_cells[ii + jj * params.nx].speeds[2] =
                  cells[ii + jj * params.nx].speeds[4];
              tmp_cells[ii + jj * params.nx].speeds[3] =
                  cells[ii + jj * params.nx].speeds[1];
              tmp_cells[ii + jj * params.nx].speeds[4] =
                  cells[ii + jj * params.nx].speeds[2];
              tmp_cells[ii + jj * params.nx].speeds[5] =
                  cells[ii + jj * params.nx].speeds[7];
              tmp_cells[ii + jj * params.nx].speeds[6] =
                  cells[ii + jj * params.nx].speeds[8];
              tmp_cells[ii + jj * params.nx].speeds[7] =
                  cells[ii + jj * params.nx].speeds[5];
              tmp_cells[ii + jj * params.nx].speeds[8] =
                  cells[ii + jj * params.nx].speeds[6];
            }
          }
        }
      }
    }
  }
}

/*
** Particles flow to the corresponding cell according to their speed
*direaction.
*/
int streaming(int start_col, int end_col, const t_param params, t_speed *cells,
              t_speed *tmp_cells) {
  /* loop over _all_ cells */

#pragma omp parallel
  {
    t_speed buffer;
    int id = omp_get_thread_num();
    int col_per_thread = params.nx / omp_get_num_threads() + 1;
    int start_col = id * col_per_thread, end_col = (id + 1) * col_per_thread;
    int jj_start, jj_end, ii_start, ii_end;

    if (end_col > params.nx)
      end_col = params.nx;
    for (int j = 0; j < params.ny; j += chunk_y) {
      jj_start = j;
      jj_end = ((j + chunk_y) > params.ny) ? params.ny : (j + chunk_y);
      for (int i = start_col; i < end_col; i += chunk_x) {
        ii_start = i;
        ii_end = ((i + chunk_x) > end_col) ? end_col : (i + chunk_x);
        for (int jj = jj_start, jj_offset = 0; jj < jj_end; jj++, jj_offset++) {
          for (int ii = ii_start, ii_offset = 0; ii < ii_end;
               ii++, ii_offset++) {
            int y_n = ((jj + 1) >= params.ny) ? 0 : jj + 1;
            int x_e = ((ii + 1) >= params.nx) ? 0 : ii + 1;
            int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);
            int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);
            /* propagate densities from neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */
            cells[ii + jj * params.nx].speeds[0] =
                tmp_cells[ii + jj * params.nx]
                    .speeds[0]; /* central cell, no movement */
            cells[x_e + jj * params.nx].speeds[1] =
                tmp_cells[ii + jj * params.nx].speeds[1]; /* east */
            cells[ii + y_n * params.nx].speeds[2] =
                tmp_cells[ii + jj * params.nx].speeds[2]; /* north */
            cells[x_w + jj * params.nx].speeds[3] =
                tmp_cells[ii + jj * params.nx].speeds[3]; /* west */
            cells[ii + y_s * params.nx].speeds[4] =
                tmp_cells[ii + jj * params.nx].speeds[4]; /* south */
            cells[x_e + y_n * params.nx].speeds[5] =
                tmp_cells[ii + jj * params.nx].speeds[5]; /* north-east */
            cells[x_w + y_n * params.nx].speeds[6] =
                tmp_cells[ii + jj * params.nx].speeds[6]; /* north-west */
            cells[x_w + y_s * params.nx].speeds[7] =
                tmp_cells[ii + jj * params.nx].speeds[7]; /* south-west */
            cells[x_e + y_s * params.nx].speeds[8] =
                tmp_cells[ii + jj * params.nx].speeds[8]; /* south-east */
          }
        }
      }
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
inline int boundary(const t_param params, t_speed *cells, t_speed *tmp_cells,
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
    cells[ii + jj * params.nx].speeds[4] =
        tmp_cells[ii + jj * params.nx].speeds[2];
    cells[ii + jj * params.nx].speeds[7] =
        tmp_cells[ii + jj * params.nx].speeds[5];
    cells[ii + jj * params.nx].speeds[8] =
        tmp_cells[ii + jj * params.nx].speeds[6];
  }

  // bottom wall (bounce)
  jj = 0;
#pragma omp parallel for
  for (ii = 0; ii < params.nx; ii++) {
    cells[ii + jj * params.nx].speeds[2] =
        tmp_cells[ii + jj * params.nx].speeds[4];
    cells[ii + jj * params.nx].speeds[5] =
        tmp_cells[ii + jj * params.nx].speeds[7];
    cells[ii + jj * params.nx].speeds[6] =
        tmp_cells[ii + jj * params.nx].speeds[8];
  }

  // left wall (inlet)
  ii = 0;
#pragma omp parallel for
  for (jj = 0; jj < params.ny; jj++) {
    local_density = (cells[ii + jj * params.nx].speeds[0] +
                     cells[ii + jj * params.nx].speeds[2] +
                     cells[ii + jj * params.nx].speeds[4] +
                     2.0 * cells[ii + jj * params.nx].speeds[3] +
                     2.0 * cells[ii + jj * params.nx].speeds[6] +
                     2.0 * cells[ii + jj * params.nx].speeds[7]) /
                    (1.0 - inlets[jj]);

    cells[ii + jj * params.nx].speeds[1] =
        cells[ii + jj * params.nx].speeds[3] +
        cst1 * local_density * inlets[jj];

    cells[ii + jj * params.nx].speeds[5] =
        cells[ii + jj * params.nx].speeds[7] -
        cst3 * (cells[ii + jj * params.nx].speeds[2] -
                cells[ii + jj * params.nx].speeds[4]) +
        cst2 * local_density * inlets[jj];

    cells[ii + jj * params.nx].speeds[8] =
        cells[ii + jj * params.nx].speeds[6] +
        cst3 * (cells[ii + jj * params.nx].speeds[2] -
                cells[ii + jj * params.nx].speeds[4]) +
        cst2 * local_density * inlets[jj];
  }

  // right wall (outlet)
  ii = params.nx - 1;
#pragma omp parallel for
  for (jj = 0; jj < params.ny; jj++) {

    for (int kk = 0; kk < NSPEEDS; kk++) {
      cells[ii + jj * params.nx].speeds[kk] =
          cells[ii - 1 + jj * params.nx].speeds[kk];
    }
  }

  return EXIT_SUCCESS;
}
