#include "d2q9_bgk.h"
#include "immintrin.h"
#include "types.h"
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
int fuse(int start_col, int end_col, const t_param params, t_speed *cells,
         t_speed *tmp_cells, float *inlets, int *obstacles);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** collision(), obstacle(), streaming() & boundary()
*/
int timestep(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets, int *obstacles) {
  /* The main time overhead, you should mainly optimize these processes. */

  // collision(0, params.nx, params, cells, tmp_cells, obstacles);
  // streaming(0, params.nx, params, cells, tmp_cells, inlets, obstacles);
  fuse(0, params.nx, params, cells, tmp_cells, inlets, obstacles);

  // boundary(params, cells, tmp_cells, inlets);

  return EXIT_SUCCESS;
}

int timestep_begin(const t_param params, t_speed *cells, t_speed *tmp_cells,
                   float *inlets, int *obstacles) {
  collision(0, params.nx, params, cells, tmp_cells, obstacles);
  return 0;
}
int timestep_end(const t_param params, t_speed *cells, t_speed *tmp_cells,
                 float *inlets, int *obstacles) {
  streaming(0, params.nx, params, cells, tmp_cells);
  boundary(params, cells, tmp_cells, inlets);
  return 0;
}

/*
** The collision of fluids in the cell is calculated using
** the local equilibrium distribution and relaxation process
*/

#define chunk_x 64
#define chunk_y 64

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

#pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
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

        __m256 x = _mm256_setr_ps(u_x, u_y, -u_x, -u_y, u_x + u_y, -u_x + u_y,
                                  -u_x - u_y, u_x - u_y);

        __m256 res = _mm256_add_ps(
            _mm256_add_ps(_1, _mm256_div_ps(x, c)),
            _mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(x, x), _2_c_c),
                          _mm256_set1_ps(u_sq / (2.f * c_sq))));
        res =
            _mm256_mul_ps(_mm256_mul_ps(res, _mm256_set1_ps(local_density)), w);
        /* relaxation step */
        tmp_cells[ii + jj * params.nx].speeds[0] =
            cells[ii + jj * params.nx].speeds[0] +
            params.omega * (d_equ - cells[ii + jj * params.nx].speeds[0]);
        __m256 c_s = _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);
        res = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(res, c_s), omega), c_s);
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

inline int streaming(int start_col, int end_col, const t_param params,
                     t_speed *cells, t_speed *tmp_cells) {
#pragma omp parallel for
  for (int i = start_col; i < end_col; i += chunk_x) {
    // for (int j = 0; j < params.ny; j += chunk_y) {
    for (int jj = 0; jj < params.ny; jj++) {
      for (int ii = i; ii < i + chunk_x; ii++) {
        int y_s = ((jj + 1) >= params.ny) ? 0 : jj + 1;
        int x_w = ((ii + 1) >= params.nx) ? 0 : ii + 1;
        int y_n = (jj == 0) ? (params.ny - 1) : (jj - 1);
        int x_e = (ii == 0) ? (params.nx - 1) : (ii - 1);
        /* propagate densities from neighbouring cells, following
        ** appropriate directions of travel and writing into
        ** scratch space grid */
        cells[ii + jj * params.nx].speeds[0] =
            tmp_cells[ii + jj * params.nx]
                .speeds[0]; /* central cell, no movement */
        cells[ii + jj * params.nx].speeds[1] =
            tmp_cells[x_e + jj * params.nx].speeds[1]; /* east */
        cells[ii + jj * params.nx].speeds[2] =
            tmp_cells[ii + y_n * params.nx].speeds[2]; /* north */
        cells[ii + jj * params.nx].speeds[3] =
            tmp_cells[x_w + jj * params.nx].speeds[3]; /* west */
        cells[ii + jj * params.nx].speeds[4] =
            tmp_cells[ii + y_s * params.nx].speeds[4]; /* south */
        cells[ii + jj * params.nx].speeds[5] =
            tmp_cells[x_e + y_n * params.nx].speeds[5]; /* north-east */
        cells[ii + jj * params.nx].speeds[6] =
            tmp_cells[x_w + y_n * params.nx].speeds[6]; /* north-west */
        cells[ii + jj * params.nx].speeds[7] =
            tmp_cells[x_w + y_s * params.nx].speeds[7]; /* south-west */
        cells[ii + jj * params.nx].speeds[8] =
            tmp_cells[x_e + y_s * params.nx].speeds[8]; /* south-east */
      }
    }
  }
}

/*
** Particles flow to the corresponding cell according to their speed
*direaction.
*/
inline int fuse(int start_col, int end_col, const t_param params,
                t_speed *cells, t_speed *tmp_cells, float *inlets,
                int *obstacles) {
  /* loop over _all_ cells */

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

#pragma omp parallel for
  for (int i = 1; i < params.nx - 2; i += chunk_x) {
    int ii_start = i, ii_end = ((i + chunk_x) > (params.nx - 1))
                                   ? (params.nx - 1)
                                   : (i + chunk_x);
    int y_s = 0;
    int y_n = 0;
    int x_w = 0;
    int x_e = 0;

    for (int jj = 1; jj < params.ny - 1; jj++) {
      y_s = jj + 1;
      y_n = jj - 1;
      for (int ii = ii_start; ii < ii_end; ii++) {
        x_w = ii + 1;
        x_e = ii - 1;
        /* propagate densities from neighbouring cells, following
        ** appropriate directions of travel and writing into
        ** scratch space grid */
        cells[ii + jj * params.nx].speeds[0] =
            tmp_cells[ii + jj * params.nx]
                .speeds[0]; /* central cell, no movement */
        cells[ii + jj * params.nx].speeds[1] =
            tmp_cells[x_e + jj * params.nx].speeds[1]; /* east */
        cells[ii + jj * params.nx].speeds[2] =
            tmp_cells[ii + y_n * params.nx].speeds[2]; /* north */
        cells[ii + jj * params.nx].speeds[3] =
            tmp_cells[x_w + jj * params.nx].speeds[3]; /* west */
        cells[ii + jj * params.nx].speeds[4] =
            tmp_cells[ii + y_s * params.nx].speeds[4]; /* south */
        cells[ii + jj * params.nx].speeds[5] =
            tmp_cells[x_e + y_n * params.nx].speeds[5]; /* north-east */
        cells[ii + jj * params.nx].speeds[6] =
            tmp_cells[x_w + y_n * params.nx].speeds[6]; /* north-west */
        cells[ii + jj * params.nx].speeds[7] =
            tmp_cells[x_w + y_s * params.nx].speeds[7]; /* south-west */
        cells[ii + jj * params.nx].speeds[8] =
            tmp_cells[x_e + y_s * params.nx].speeds[8]; /* south-east */

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

          __m256 x = _mm256_setr_ps(u_x, u_y, -u_x, -u_y, u_x + u_y, -u_x + u_y,
                                    -u_x - u_y, u_x - u_y);

          __m256 res = _mm256_add_ps(
              _mm256_add_ps(_1, _mm256_div_ps(x, c)),
              _mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(x, x),
                                          _mm256_set1_ps(2.f * c_sq * c_sq)),
                            _mm256_set1_ps(u_sq / (2.f * c_sq))));
          res = _mm256_mul_ps(_mm256_mul_ps(res, _mm256_set1_ps(local_density)),
                              _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2));
          /* relaxation step */
          cells[ii + jj * params.nx].speeds[0] =
              cells[ii + jj * params.nx].speeds[0] +
              params.omega * (d_equ - cells[ii + jj * params.nx].speeds[0]);
          __m256 c_s = _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);
          res =
              _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(res, c_s), omega), c_s);
          _mm256_storeu_ps(cells[ii + jj * params.nx].speeds + 1, res);
        } else {
          float tmp;
          tmp = cells[ii + jj * params.nx].speeds[3];
          cells[ii + jj * params.nx].speeds[3] =
              cells[ii + jj * params.nx].speeds[1];
          cells[ii + jj * params.nx].speeds[1] = tmp;

          tmp = cells[ii + jj * params.nx].speeds[4];
          cells[ii + jj * params.nx].speeds[4] =
              cells[ii + jj * params.nx].speeds[2];
          cells[ii + jj * params.nx].speeds[2] = tmp;

          tmp = cells[ii + jj * params.nx].speeds[5];
          cells[ii + jj * params.nx].speeds[5] =
              cells[ii + jj * params.nx].speeds[7];
          cells[ii + jj * params.nx].speeds[7] = tmp;

          tmp = cells[ii + jj * params.nx].speeds[6];
          cells[ii + jj * params.nx].speeds[6] =
              cells[ii + jj * params.nx].speeds[8];
          cells[ii + jj * params.nx].speeds[8] = tmp;
        }
      }
    }
  }
  int jj = 0;
  int y_s = jj + 1;
  int y_n = params.ny - 1;
  int x_w = 0;
  int x_e = 0;
#pragma omp parallel for
  for (int ii = 1; ii < params.nx - 2; ii++) {
    x_w = ii + 1;
    x_e = ii - 1;
    // ! stream
    cells[ii + jj * params.nx].speeds[0] =
        tmp_cells[ii + jj * params.nx]
            .speeds[0]; /* central cell, no movement */
    cells[ii + jj * params.nx].speeds[1] =
        tmp_cells[x_e + jj * params.nx].speeds[1]; /* east */

    cells[ii + jj * params.nx].speeds[3] =
        tmp_cells[x_w + jj * params.nx].speeds[3]; /* west */
    cells[ii + jj * params.nx].speeds[4] =
        tmp_cells[ii + y_s * params.nx].speeds[4]; /* south */

    cells[ii + jj * params.nx].speeds[7] =
        tmp_cells[x_w + y_s * params.nx].speeds[7]; /* south-west */
    cells[ii + jj * params.nx].speeds[8] =
        tmp_cells[x_e + y_s * params.nx].speeds[8]; /* south-east */
    // ! boundary
    cells[ii + jj * params.nx].speeds[2] =
        tmp_cells[ii + jj * params.nx].speeds[4];
    cells[ii + jj * params.nx].speeds[5] =
        tmp_cells[ii + jj * params.nx].speeds[7];
    cells[ii + jj * params.nx].speeds[6] =
        tmp_cells[ii + jj * params.nx].speeds[8];
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

      __m256 x = _mm256_setr_ps(u_x, u_y, -u_x, -u_y, u_x + u_y, -u_x + u_y,
                                -u_x - u_y, u_x - u_y);

      __m256 res = _mm256_add_ps(
          _mm256_add_ps(_1, _mm256_div_ps(x, c)),
          _mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(x, x),
                                      _mm256_set1_ps(2.f * c_sq * c_sq)),
                        _mm256_set1_ps(u_sq / (2.f * c_sq))));
      res = _mm256_mul_ps(_mm256_mul_ps(res, _mm256_set1_ps(local_density)),
                          _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2));
      /* relaxation step */
      cells[ii + jj * params.nx].speeds[0] =
          cells[ii + jj * params.nx].speeds[0] +
          params.omega * (d_equ - cells[ii + jj * params.nx].speeds[0]);
      __m256 c_s = _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);
      res = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(res, c_s), omega), c_s);
      _mm256_storeu_ps(cells[ii + jj * params.nx].speeds + 1, res);
    } else {
      float tmp;
      tmp = cells[ii + jj * params.nx].speeds[3];
      cells[ii + jj * params.nx].speeds[3] =
          cells[ii + jj * params.nx].speeds[1];
      cells[ii + jj * params.nx].speeds[1] = tmp;

      tmp = cells[ii + jj * params.nx].speeds[4];
      cells[ii + jj * params.nx].speeds[4] =
          cells[ii + jj * params.nx].speeds[2];
      cells[ii + jj * params.nx].speeds[2] = tmp;

      tmp = cells[ii + jj * params.nx].speeds[5];
      cells[ii + jj * params.nx].speeds[5] =
          cells[ii + jj * params.nx].speeds[7];
      cells[ii + jj * params.nx].speeds[7] = tmp;

      tmp = cells[ii + jj * params.nx].speeds[6];
      cells[ii + jj * params.nx].speeds[6] =
          cells[ii + jj * params.nx].speeds[8];
      cells[ii + jj * params.nx].speeds[8] = tmp;
    }
  }

  jj = params.ny - 1;
  y_s = 0;
  y_n = jj - 1;
#pragma omp parallel for
  for (int ii = 1; ii < params.nx - 2; ii++) {
    x_w = ii + 1;
    x_e = ii - 1;
    // ! stream
    cells[ii + jj * params.nx].speeds[0] =
        tmp_cells[ii + jj * params.nx]
            .speeds[0]; /* central cell, no movement */
    cells[ii + jj * params.nx].speeds[1] =
        tmp_cells[x_e + jj * params.nx].speeds[1]; /* east */
    cells[ii + jj * params.nx].speeds[2] =
        tmp_cells[ii + y_n * params.nx].speeds[2]; /* north */
    cells[ii + jj * params.nx].speeds[3] =
        tmp_cells[x_w + jj * params.nx].speeds[3]; /* west */

    cells[ii + jj * params.nx].speeds[5] =
        tmp_cells[x_e + y_n * params.nx].speeds[5]; /* north-east */
    cells[ii + jj * params.nx].speeds[6] =
        tmp_cells[x_w + y_n * params.nx].speeds[6]; /* north-west */

    // ! boundary
    cells[ii + jj * params.nx].speeds[4] =
        tmp_cells[ii + jj * params.nx].speeds[2];
    cells[ii + jj * params.nx].speeds[7] =
        tmp_cells[ii + jj * params.nx].speeds[5];
    cells[ii + jj * params.nx].speeds[8] =
        tmp_cells[ii + jj * params.nx].speeds[6];

    // ! collision
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

      __m256 x = _mm256_setr_ps(u_x, u_y, -u_x, -u_y, u_x + u_y, -u_x + u_y,
                                -u_x - u_y, u_x - u_y);

      __m256 res = _mm256_add_ps(
          _mm256_add_ps(_1, _mm256_div_ps(x, c)),
          _mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(x, x),
                                      _mm256_set1_ps(2.f * c_sq * c_sq)),
                        _mm256_set1_ps(u_sq / (2.f * c_sq))));
      res = _mm256_mul_ps(_mm256_mul_ps(res, _mm256_set1_ps(local_density)),
                          _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2));
      /* relaxation step */
      cells[ii + jj * params.nx].speeds[0] =
          cells[ii + jj * params.nx].speeds[0] +
          params.omega * (d_equ - cells[ii + jj * params.nx].speeds[0]);
      __m256 c_s = _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);
      res = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(res, c_s), omega), c_s);
      _mm256_storeu_ps(cells[ii + jj * params.nx].speeds + 1, res);
    } else {
      float tmp;
      tmp = cells[ii + jj * params.nx].speeds[3];
      cells[ii + jj * params.nx].speeds[3] =
          cells[ii + jj * params.nx].speeds[1];
      cells[ii + jj * params.nx].speeds[1] = tmp;

      tmp = cells[ii + jj * params.nx].speeds[4];
      cells[ii + jj * params.nx].speeds[4] =
          cells[ii + jj * params.nx].speeds[2];
      cells[ii + jj * params.nx].speeds[2] = tmp;

      tmp = cells[ii + jj * params.nx].speeds[5];
      cells[ii + jj * params.nx].speeds[5] =
          cells[ii + jj * params.nx].speeds[7];
      cells[ii + jj * params.nx].speeds[7] = tmp;

      tmp = cells[ii + jj * params.nx].speeds[6];
      cells[ii + jj * params.nx].speeds[6] =
          cells[ii + jj * params.nx].speeds[8];
      cells[ii + jj * params.nx].speeds[8] = tmp;
    }
  }

  const float cst1 = 2.0 / 3.0;
  const float cst2 = 1.0 / 6.0;
  const float cst3 = 1.0 / 2.0;

  float local_density;

  int ii = 0;
#pragma omp parallel for
  for (int jj = 0; jj < params.ny; ++jj) {
    y_s = (jj + 1) >= params.ny ? 0 : (jj + 1);
    y_n = (jj - 1) < 0 ? params.ny - 1 : (jj - 1);
    x_w = ii + 1;
    x_e = params.nx - 1;
    cells[ii + jj * params.nx].speeds[0] =
        tmp_cells[ii + jj * params.nx]
            .speeds[0]; /* central cell, no movement */
    cells[ii + jj * params.nx].speeds[1] =
        tmp_cells[x_e + jj * params.nx].speeds[1]; /* east */
    cells[ii + jj * params.nx].speeds[2] =
        tmp_cells[ii + y_n * params.nx].speeds[2]; /* north */
    cells[ii + jj * params.nx].speeds[3] =
        tmp_cells[x_w + jj * params.nx].speeds[3]; /* west */
    cells[ii + jj * params.nx].speeds[4] =
        tmp_cells[ii + y_s * params.nx].speeds[4]; /* south */
    cells[ii + jj * params.nx].speeds[5] =
        tmp_cells[x_e + y_n * params.nx].speeds[5]; /* north-east */
    cells[ii + jj * params.nx].speeds[6] =
        tmp_cells[x_w + y_n * params.nx].speeds[6]; /* north-west */
    cells[ii + jj * params.nx].speeds[7] =
        tmp_cells[x_w + y_s * params.nx].speeds[7]; /* south-west */
    cells[ii + jj * params.nx].speeds[8] =
        tmp_cells[x_e + y_s * params.nx].speeds[8]; /* south-east */

    if (jj == 0) {
      cells[ii + jj * params.nx].speeds[2] =
          tmp_cells[ii + jj * params.nx].speeds[4];
      cells[ii + jj * params.nx].speeds[5] =
          tmp_cells[ii + jj * params.nx].speeds[7];
      cells[ii + jj * params.nx].speeds[6] =
          tmp_cells[ii + jj * params.nx].speeds[8];
    } else if (jj == params.ny - 1) {
      cells[ii + jj * params.nx].speeds[4] =
          tmp_cells[ii + jj * params.nx].speeds[2];
      cells[ii + jj * params.nx].speeds[7] =
          tmp_cells[ii + jj * params.nx].speeds[5];
      cells[ii + jj * params.nx].speeds[8] =
          tmp_cells[ii + jj * params.nx].speeds[6];
    }

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

    // ! collision
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

      __m256 x = _mm256_setr_ps(u_x, u_y, -u_x, -u_y, u_x + u_y, -u_x + u_y,
                                -u_x - u_y, u_x - u_y);

      __m256 res = _mm256_add_ps(
          _mm256_add_ps(_1, _mm256_div_ps(x, c)),
          _mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(x, x),
                                      _mm256_set1_ps(2.f * c_sq * c_sq)),
                        _mm256_set1_ps(u_sq / (2.f * c_sq))));
      res = _mm256_mul_ps(_mm256_mul_ps(res, _mm256_set1_ps(local_density)),
                          _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2));
      /* relaxation step */
      cells[ii + jj * params.nx].speeds[0] =
          cells[ii + jj * params.nx].speeds[0] +
          params.omega * (d_equ - cells[ii + jj * params.nx].speeds[0]);
      __m256 c_s = _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);
      res = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(res, c_s), omega), c_s);
      _mm256_storeu_ps(cells[ii + jj * params.nx].speeds + 1, res);
    } else {
      float tmp;
      tmp = cells[ii + jj * params.nx].speeds[3];
      cells[ii + jj * params.nx].speeds[3] =
          cells[ii + jj * params.nx].speeds[1];
      cells[ii + jj * params.nx].speeds[1] = tmp;

      tmp = cells[ii + jj * params.nx].speeds[4];
      cells[ii + jj * params.nx].speeds[4] =
          cells[ii + jj * params.nx].speeds[2];
      cells[ii + jj * params.nx].speeds[2] = tmp;

      tmp = cells[ii + jj * params.nx].speeds[5];
      cells[ii + jj * params.nx].speeds[5] =
          cells[ii + jj * params.nx].speeds[7];
      cells[ii + jj * params.nx].speeds[7] = tmp;

      tmp = cells[ii + jj * params.nx].speeds[6];
      cells[ii + jj * params.nx].speeds[6] =
          cells[ii + jj * params.nx].speeds[8];
      cells[ii + jj * params.nx].speeds[8] = tmp;
    }
  }

  ii = params.nx - 1;
  t_speed prev_cell;
#pragma omp parallel for private(prev_cell)
  for (int jj = 0; jj < params.ny; ++jj) {
    for (int ii_offset = 0; ii_offset < 2; ++ii_offset) {
      ii = ii_offset + params.nx - 2;
      y_s = (jj + 1) >= params.ny ? 0 : (jj + 1);
      y_n = (jj - 1) < 0 ? params.ny - 1 : (jj - 1);
      x_w = (ii_offset == 0) ? ii + 1 : 0;
      x_e = ii - 1;
      cells[ii + jj * params.nx].speeds[0] =
          tmp_cells[ii + jj * params.nx]
              .speeds[0]; /* central cell, no movement */
      cells[ii + jj * params.nx].speeds[1] =
          tmp_cells[x_e + jj * params.nx].speeds[1]; /* east */
      cells[ii + jj * params.nx].speeds[2] =
          tmp_cells[ii + y_n * params.nx].speeds[2]; /* north */
      cells[ii + jj * params.nx].speeds[3] =
          tmp_cells[x_w + jj * params.nx].speeds[3]; /* west */
      cells[ii + jj * params.nx].speeds[4] =
          tmp_cells[ii + y_s * params.nx].speeds[4]; /* south */
      cells[ii + jj * params.nx].speeds[5] =
          tmp_cells[x_e + y_n * params.nx].speeds[5]; /* north-east */
      cells[ii + jj * params.nx].speeds[6] =
          tmp_cells[x_w + y_n * params.nx].speeds[6]; /* north-west */
      cells[ii + jj * params.nx].speeds[7] =
          tmp_cells[x_w + y_s * params.nx].speeds[7]; /* south-west */
      cells[ii + jj * params.nx].speeds[8] =
          tmp_cells[x_e + y_s * params.nx].speeds[8]; /* south-east */

      if (jj == 0) {
        cells[ii + jj * params.nx].speeds[2] =
            tmp_cells[ii + jj * params.nx].speeds[4];
        cells[ii + jj * params.nx].speeds[5] =
            tmp_cells[ii + jj * params.nx].speeds[7];
        cells[ii + jj * params.nx].speeds[6] =
            tmp_cells[ii + jj * params.nx].speeds[8];
      } else if (jj == params.ny - 1) {
        cells[ii + jj * params.nx].speeds[4] =
            tmp_cells[ii + jj * params.nx].speeds[2];
        cells[ii + jj * params.nx].speeds[7] =
            tmp_cells[ii + jj * params.nx].speeds[5];
        cells[ii + jj * params.nx].speeds[8] =
            tmp_cells[ii + jj * params.nx].speeds[6];
      }

      if (ii_offset == 0)
        prev_cell = cells[ii + jj * params.nx];
      else {
        for (int kk = 0; kk < NSPEEDS; kk++) {
          cells[ii + jj * params.nx].speeds[kk] = prev_cell.speeds[kk];
        }
      }

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

        __m256 x = _mm256_setr_ps(u_x, u_y, -u_x, -u_y, u_x + u_y, -u_x + u_y,
                                  -u_x - u_y, u_x - u_y);

        __m256 res = _mm256_add_ps(
            _mm256_add_ps(_1, _mm256_div_ps(x, c)),
            _mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(x, x),
                                        _mm256_set1_ps(2.f * c_sq * c_sq)),
                          _mm256_set1_ps(u_sq / (2.f * c_sq))));
        res = _mm256_mul_ps(_mm256_mul_ps(res, _mm256_set1_ps(local_density)),
                            _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2));
        /* relaxation step */
        cells[ii + jj * params.nx].speeds[0] =
            cells[ii + jj * params.nx].speeds[0] +
            params.omega * (d_equ - cells[ii + jj * params.nx].speeds[0]);
        __m256 c_s = _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);
        res = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(res, c_s), omega), c_s);
        _mm256_storeu_ps(cells[ii + jj * params.nx].speeds + 1, res);
      } else {
        float tmp;
        tmp = cells[ii + jj * params.nx].speeds[3];
        cells[ii + jj * params.nx].speeds[3] =
            cells[ii + jj * params.nx].speeds[1];
        cells[ii + jj * params.nx].speeds[1] = tmp;

        tmp = cells[ii + jj * params.nx].speeds[4];
        cells[ii + jj * params.nx].speeds[4] =
            cells[ii + jj * params.nx].speeds[2];
        cells[ii + jj * params.nx].speeds[2] = tmp;

        tmp = cells[ii + jj * params.nx].speeds[5];
        cells[ii + jj * params.nx].speeds[5] =
            cells[ii + jj * params.nx].speeds[7];
        cells[ii + jj * params.nx].speeds[7] = tmp;

        tmp = cells[ii + jj * params.nx].speeds[6];
        cells[ii + jj * params.nx].speeds[6] =
            cells[ii + jj * params.nx].speeds[8];
        cells[ii + jj * params.nx].speeds[8] = tmp;
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
