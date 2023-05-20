/* This project is based on COMS30006 - Advanced High Performance Computing
   - Lattice Boltzmann in the University of Bristol. The project is been
   modified to solve open boundaries for pipe flow and be more friendly
   to CS110 students in ShanghaiTech University by Lei Jia in 2023.
*/

#include "aa.h"
#include "aa_AoS.h"
#include "calc.h"
#include "d2q9_bgk.h"
#include "types.h"
#include "utils.h"

/* output usage examples */
void usage(const char *exe) {
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile> [output_directory]\n",
          exe);
  exit(EXIT_FAILURE);
}

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char *argv[]) {
  char *paramfile = NULL;    /* name of the input parameter file */
  char *obstaclefile = NULL; /* name of a the input obstacle file */
  char *out_dir = NULL;      /* name of output directory */
  t_param params;            /* struct to hold parameter values */
  t_speed *cells = NULL;     /* grid containing fluid densities */
  t_speed *tmp_cells = NULL; /* scratch space */
  int *obstacles = NULL;     /* grid indicating which cells are blocked */
  float *inlets = NULL;      /* inlet velocity */
  struct timeval timstr;     /* structure to hold elapsed time */
  double total_time, init_time,
      comp_time; /* floating point numbers to calculate elapsed wallclock time
                  */
  char buf[128]; /* a string buffer for specific filename */

  /* parse the command line */
  if (argc < 3) {
    usage(argv[0]);
  } else {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }
  /* set the output directory */
  if (argc >= 4)
    out_dir = argv[3];
  else
    out_dir = "./results";

  /* Display load parameters */
  printf("==load==\n");
  printf("Params file:   %s\n", paramfile);
  printf("Obstacle file: %s\n", obstaclefile);
  printf("Out directory: %s\n", out_dir);
  /* Total/init time starts here */
  gettimeofday(&timstr, NULL);
  total_time = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_time = total_time;
  /* initialise our data structures and load values from file */

  // ! cells_aos_ptr
  t_speed_aos *cells_aos;
  // ! type 0 for SoA 1 for AoS
  int type = 0;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles,
             &inlets, &cells_aos, &type);

  /* Set the inlet speed */
  set_inlets(params, inlets);
  /* Init time stops here */
  gettimeofday(&timstr, NULL);
  init_time = timstr.tv_sec + (timstr.tv_usec / 1000000.0) - init_time;

  /* Display simulation parameters */
  printf("==start==\n");
  printf("Number of cells:\t\t\t%d (%d x %d)\n", params.nx * params.ny,
         params.nx, params.ny);
  printf("Max iterations:\t\t\t\t%d\n", params.maxIters);
  printf("Density:\t\t\t\t%.6lf\n", params.density);
  printf("Kinematic viscosity:\t\t\t%.6lf\n", params.viscosity);
  printf("Inlet velocity:\t\t\t\t%.6lf\n", params.velocity);
  printf("Inlet type:\t\t\t\t%d\n", params.type);
  printf("Relaxtion parameter:\t\t\t%.6lf\n", params.omega);

  /* Compute time starts */
  gettimeofday(&timstr, NULL);
  comp_time = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* timestep loop */
  for (int tt = 0; tt < params.maxIters; tt += 2) {
    // timestep(params, cells, tmp_cells, inlets, obstacles);
    if (type == 0) {
      aa_odd_timestep(params, cells, tmp_cells, inlets, obstacles);
      aa_even_timestep(params, cells, tmp_cells, inlets, obstacles);
    } else if (type == 1) {
      AoS_aa_odd_timestep(params, cells_aos, tmp_cells, inlets, obstacles);
      AoS_aa_even_timestep(params, cells_aos, tmp_cells, inlets, obstacles);
    }

/* Visualization */
#ifdef VISUAL
    if (tt % 1000 == 0) {
      sprintf(buf, "%s/visual/state_%d.dat", out_dir, tt / 1000);
      write_state(buf, params, cells, obstacles, cells_aos, type);
    }
#endif
  }

  /* Compute time stops here */
  gettimeofday(&timstr, NULL);
  comp_time = timstr.tv_sec + (timstr.tv_usec / 1000000.0) - comp_time;

  /* write final state and free memory */
  sprintf(buf, "%s/final_state.dat", out_dir);
  write_state(buf, params, cells, obstacles, cells_aos, type);

  /* Display Reynolds number and time */
  printf("==done==\n");
  printf("Reynolds number:\t\t\t%.12E\n",
         calc_reynolds(params, cells, obstacles, cells_aos, type));
  printf("Average velocity:\t\t\t%.12E\n",
         av_velocity(params, cells, obstacles, cells_aos, type));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n", init_time);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_time);

  /* finalise */
  finalise(&params, &cells, &tmp_cells, &obstacles, &inlets, &cells_aos, type);
  /* total time stop */
  gettimeofday(&timstr, NULL);
  total_time = timstr.tv_sec + (timstr.tv_usec / 1000000.0) - total_time;
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n", total_time);

  return EXIT_SUCCESS;
}
