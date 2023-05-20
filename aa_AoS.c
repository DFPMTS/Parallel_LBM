#pragma once
#include "aa_AoS.h"
#include "immintrin.h"
#include "types.h"
#include <omp.h>
#include <stdio.h>
#include <xmmintrin.h>

static float top[4096][3];
static float down[4096][3];

int AoS_aa_even_timestep(const t_param params, t_speed_aos *cells,
                         t_speed *tmp_cells, float *inlets, int *obstacles) {
  AoS_aa_even(params, cells, tmp_cells, obstacles, inlets);
  AoS_aa_boundary(params, cells, tmp_cells, inlets);
  return 0;
}

int AoS_aa_odd_timestep(const t_param params, t_speed_aos *cells,
                        t_speed *tmp_cells, float *inlets, int *obstacles) {
  AoS_aa_odd(params, cells, tmp_cells, obstacles);
  return 0;
}

static float buffer[9];

int AoS_aa_odd(const t_param params, t_speed_aos *cells, t_speed *tmp_cells,
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

  // ! jj = 0  -----------------------------------

  int jj = 0;
#pragma omp parallel for private(buffer)
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
      __m256 c_s = _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);

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
      res = _mm256_mul_ps(_mm256_mul_ps(res, _mm256_set1_ps(local_density)), w);
      /* relaxation step */
      cells[ii + jj * params.nx].speeds[0] =
          cells[ii + jj * params.nx].speeds[0] +
          params.omega * (d_equ - cells[ii + jj * params.nx].speeds[0]);

      res = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(res, c_s), omega), c_s);
      _mm256_storeu_ps(cells[ii + jj * params.nx].speeds + 1,
                       _mm256_permute_ps(res, 0b01001110));

      // cells[ii + jj * params.nx].speeds[1] = buffer[3];
      // cells[ii + jj * params.nx].speeds[3] = buffer[1];
      // cells[ii + jj * params.nx].speeds[2] = buffer[4];
      // cells[ii + jj * params.nx].speeds[4] = buffer[2];
      // cells[ii + jj * params.nx].speeds[5] = buffer[7];
      // cells[ii + jj * params.nx].speeds[7] = buffer[5];
      // cells[ii + jj * params.nx].speeds[6] = buffer[8];
      // cells[ii + jj * params.nx].speeds[8] = buffer[6];
    }
    down[ii][0] = cells[ii + jj * params.nx].speeds[2];
    down[ii][1] = cells[ii + jj * params.nx].speeds[5];
    down[ii][2] = cells[ii + jj * params.nx].speeds[6];
  }

  // ! jj = param.ny - 1 -----------------------------------

  jj = params.ny - 1;
#pragma omp parallel for private(buffer)
  for (int ii = 0; ii < params.nx; ++ii) {
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
      __m256 c_s = _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);

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
      res = _mm256_mul_ps(_mm256_mul_ps(res, _mm256_set1_ps(local_density)), w);
      /* relaxation step */
      cells[ii + jj * params.nx].speeds[0] =
          cells[ii + jj * params.nx].speeds[0] +
          params.omega * (d_equ - cells[ii + jj * params.nx].speeds[0]);

      res = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(res, c_s), omega), c_s);
      // _mm256_storeu_ps(buffer + 1, res);
      _mm256_storeu_ps(cells[ii + jj * params.nx].speeds + 1,
                       _mm256_permute_ps(res, 0b01001110));

      // cells[ii + jj * params.nx].speeds[1] = buffer[3];
      // cells[ii + jj * params.nx].speeds[3] = buffer[1];
      // cells[ii + jj * params.nx].speeds[2] = buffer[4];
      // cells[ii + jj * params.nx].speeds[4] = buffer[2];
      // cells[ii + jj * params.nx].speeds[5] = buffer[7];
      // cells[ii + jj * params.nx].speeds[7] = buffer[5];
      // cells[ii + jj * params.nx].speeds[6] = buffer[8];
      // cells[ii + jj * params.nx].speeds[8] = buffer[6];
    }
    top[ii][0] = cells[ii + jj * params.nx].speeds[4];
    top[ii][1] = cells[ii + jj * params.nx].speeds[7];
    top[ii][2] = cells[ii + jj * params.nx].speeds[8];
  }

#pragma omp parallel for private(buffer)
  for (int jj = 1; jj < params.ny - 1; jj++) {
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
        __m256 c_s = _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);

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
        cells[ii + jj * params.nx].speeds[0] =
            cells[ii + jj * params.nx].speeds[0] +
            params.omega * (d_equ - cells[ii + jj * params.nx].speeds[0]);

        res = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(res, c_s), omega), c_s);
        // _mm256_storeu_ps(buffer + 1, res);
        _mm256_storeu_ps(cells[ii + jj * params.nx].speeds + 1,
                         _mm256_permute_ps(res, 0b01001110));

        // cells[ii + jj * params.nx].speeds[1] = buffer[3];
        // cells[ii + jj * params.nx].speeds[3] = buffer[1];
        // cells[ii + jj * params.nx].speeds[2] = buffer[4];
        // cells[ii + jj * params.nx].speeds[4] = buffer[2];
        // cells[ii + jj * params.nx].speeds[5] = buffer[7];
        // cells[ii + jj * params.nx].speeds[7] = buffer[5];
        // cells[ii + jj * params.nx].speeds[6] = buffer[8];
        // cells[ii + jj * params.nx].speeds[8] = buffer[6];
      }
    }
  }
  return EXIT_SUCCESS;
}

int AoS_aa_even(const t_param params, t_speed_aos *cells, t_speed *tmp_cells,
                int *obstacles, float *inlets) {
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */

  const float cst1 = 2.0 / 3.0;
  const float cst2 = 1.0 / 6.0;
  const float cst3 = 1.0 / 2.0;

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
    float buffer[9];
    float prev[9];
    int y_n, y_s, x_e, x_w;
    float local_density;
#pragma omp for
    for (int i = 0; i < params.nx; i += 64)
      for (int jj = 0; jj < params.ny; jj++) {
        y_n = (jj + 1 == params.ny) ? 0 : (jj + 1);
        y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);
        for (int ii = i; ii < i + 128; ii++) {
          /* determine indices of axis-direction neighbours
          ** respecting periodic boundary conditions (wrap around) */
          x_e = (ii + 1 == params.nx) ? 0 : (ii + 1);
          x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);
          /* propagate densities from neighbouring cells, following
          ** appropriate directions of travel and writing into
          ** scratch space grid */

          // ! streaming
          // ------------------------------------------------------

          buffer[0] = cells[ii + jj * params.nx].speeds[0]; /* central*/

          buffer[1] = cells[x_w + jj * params.nx].speeds[3]; /* east */
          buffer[3] = cells[x_e + jj * params.nx].speeds[1]; /* west */

          buffer[2] = cells[ii + y_s * params.nx].speeds[4]; /* south */
          buffer[4] = cells[ii + y_n * params.nx].speeds[2]; /* north */

          buffer[6] = cells[x_e + y_s * params.nx].speeds[8]; /* north-west */
          buffer[8] = cells[x_w + y_n * params.nx].speeds[6]; /* south-east */

          buffer[5] = cells[x_w + y_s * params.nx].speeds[7]; /* north-east */
          buffer[7] = cells[x_e + y_n * params.nx].speeds[5]; /* north-east */

          // ! boundary  -----------------------------------------------------

          // ! load from previous pre-streaming values
          if (jj == params.ny - 1) {
            buffer[4] = top[ii][0];
            buffer[7] = top[ii][1];
            buffer[8] = top[ii][2];
          } else if (jj == 0) {
            buffer[2] = down[ii][0];
            buffer[5] = down[ii][1];
            buffer[6] = down[ii][2];
          }

          if (ii == 0) {
            float local_density =
                (buffer[0] + buffer[2] + buffer[4] + 2.0 * buffer[3] +
                 2.0 * buffer[6] + 2.0 * buffer[7]) /
                (1.0 - inlets[jj]);

            buffer[1] = buffer[3] + cst1 * local_density * inlets[jj];

            buffer[5] = buffer[7] - cst3 * (buffer[2] - buffer[4]) +
                        cst2 * local_density * inlets[jj];

            buffer[8] = buffer[6] + cst3 * (buffer[2] - buffer[4]) +
                        cst2 * local_density * inlets[jj];
          } else if (ii == params.nx - 2) {
            for (int k = 0; k < 9; ++k)
              prev[k] = buffer[k];
          } else if (ii == params.nx - 1) {
            for (int k = 0; k < 9; ++k)
              buffer[k] = prev[k];
          }

          // ! collision -----------------------------------------------------

          if (!obstacles[ii + jj * params.nx]) {
            float local_density = 0.f;

            for (int kk = 0; kk < NSPEEDS; kk++) {
              local_density += buffer[kk];
            }

            /* compute x velocity component */
            float u_x = (buffer[1] + buffer[5] + buffer[8] -
                         (buffer[3] + buffer[6] + buffer[7])) /
                        local_density;
            /* compute y velocity component */
            float u_y = (buffer[2] + buffer[5] + buffer[6] -
                         (buffer[4] + buffer[7] + buffer[8])) /
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
                _mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(x, x), _2_c_c),
                              _mm256_set1_ps(u_sq / (2.f * c_sq))));
            res = _mm256_mul_ps(
                _mm256_mul_ps(res, _mm256_set1_ps(local_density)), w);
            /* relaxation step */
            buffer[0] = buffer[0] + params.omega * (d_equ - buffer[0]);
            __m256 c_s =
                _mm256_setr_ps(buffer[1], buffer[2], buffer[3], buffer[4],
                               buffer[5], buffer[6], buffer[7], buffer[8]);

            res = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(res, c_s), omega),
                                c_s);
            _mm256_storeu_ps(buffer + 1, res);
          } else {
            float tmp;
            tmp = buffer[3];
            buffer[3] = buffer[1];
            buffer[1] = tmp;

            tmp = buffer[2];
            buffer[2] = buffer[4];
            buffer[4] = tmp;

            tmp = buffer[5];
            buffer[5] = buffer[7];
            buffer[7] = tmp;

            tmp = buffer[6];
            buffer[6] = buffer[8];
            buffer[8] = tmp;
          }

          // ! save pre-streaming values for boundary

          if (jj == 0) {
            down[ii][0] = buffer[4];
            down[ii][1] = buffer[7];
            down[ii][2] = buffer[8];
          } else if (jj == params.ny - 1) {
            top[ii][0] = buffer[2];
            top[ii][1] = buffer[5];
            top[ii][2] = buffer[6];
          }

          // ! streaming ---------------------------------------------------

          cells[ii + jj * params.nx].speeds[0] = buffer[0]; /* central*/

          cells[x_w + jj * params.nx].speeds[3] = buffer[3]; /* east */
          cells[x_e + jj * params.nx].speeds[1] = buffer[1]; /* west */

          cells[ii + y_s * params.nx].speeds[4] = buffer[4]; /* south */
          cells[ii + y_n * params.nx].speeds[2] = buffer[2]; /* north */

          cells[x_e + y_s * params.nx].speeds[8] = buffer[8]; /* north-west */
          cells[x_w + y_n * params.nx].speeds[6] = buffer[6]; /* south-east */

          cells[x_w + y_s * params.nx].speeds[7] = buffer[7]; /* north-east */
          cells[x_e + y_n * params.nx].speeds[5] = buffer[5]; /* north-east */

          // ! boundary  -----------------------------------------------------
        }
      }
  }
}

int AoS_aa_boundary(const t_param params, t_speed_aos *cells,
                    t_speed *tmp_cells, float *inlets) {
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
    cells[ii + jj * params.nx].speeds[4] = top[ii][0];
    cells[ii + jj * params.nx].speeds[7] = top[ii][1];
    cells[ii + jj * params.nx].speeds[8] = top[ii][2];
  }

  // bottom wall (bounce)
  jj = 0;
#pragma omp parallel for
  for (ii = 0; ii < params.nx; ii++) {
    cells[ii + jj * params.nx].speeds[2] = down[ii][0];
    cells[ii + jj * params.nx].speeds[5] = down[ii][1];
    cells[ii + jj * params.nx].speeds[6] = down[ii][2];
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
