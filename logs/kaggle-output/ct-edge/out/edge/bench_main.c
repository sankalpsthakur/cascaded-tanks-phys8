// step-only timing harness: precomputed inputs, no libm call inside the timed loop. Build: cc -O2 -std=gnu99 -o ct_bench bench_main.c -lm
#include "ct_step.c"
#include <stdlib.h>
int main(int argc, char **argv) {
  enum { N = 1024 };
  const int reps = argc > 1 ? atoi(argv[1]) : 20000;
  static double u[N];
  for (int t = 0; t < N; t++) u[t] = 3.0 + 2.0 * sin(0.013 * t);
  const ct_state s0 = { 3.739683641629819, 3.3841867891864097, 3.0, 3.0 };
  double acc = 0.0;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int r = 0; r < reps; r++) {
    ct_state s = s0;
    for (int t = 0; t < N; t++) { ct_step(&s, u[t]); acc += s.x2; }
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / ((double)reps * N);
  printf("%.3f ns_per_step (step only, checksum %.6f)\n", ns, acc);
  return 0;
}
