// Auto-generated Cascaded Tanks step (frozen parameters). C99, double precision.
#include <math.h>
#include <stdio.h>
#include <time.h>
#define EPS 0.0001
#define XM1 10.0
#define YCLIP 10.0
#define NSUB 4
static const double K1=0.1591886930325672, R2=0.7247611928106492, K3=0.1257444787319596, K4=0.25083597790754303, BETA=0.6402790962905923;
static const double A=1.8094653367806053, B4=1.490308843813169, P1=0.5, P3=0.5, C2=0.0, CA=-0.17331792449042163, QOV=1.0, C4=0.0, CA1=0.0;
static const double W0=1.0, W1=0.0, W2=0.0;
typedef struct { double x1, x2, um1, um2; } ct_state;   // um1,um2: previous inputs (input FIR)
static inline void rhs(double x1, double x2, double u, double *d1, double *d2) {
  x1 = x1 > 0 ? x1 : 0; x2 = x2 > 0 ? x2 : 0;
  double q1 = K1 * sqrt((x1 + EPS));
  double cap = 1.0 + CA * (x2 - 5.0); cap = cap > 0.2 ? cap : 0.2;
  double uu = u - B4; 
  double cap1 = 1.0 + CA1 * (x1 - 5.0); cap1 = cap1 > 0.2 ? cap1 : 0.2;
  *d1 = (K4 * (uu + C4 * uu * uu) - q1) / cap1;
  *d2 = (R2 * q1 - K3 * sqrt((x2 + EPS))) / cap;
}
static inline void clampx(double *x1, double *x2) {
  double sur = *x1 - XM1; sur = sur > 0 ? sur : 0;
  const double hh = 1.0 / NSUB;
  *x1 -= sur; *x2 += BETA * sur;
  double mx = YCLIP - A; if (*x2 > mx) *x2 = mx;
  if (*x1 < 0) *x1 = 0; if (*x2 < 0) *x2 = 0;
}
// advance one sample with zero-order-hold input u; returns nothing, updates state
static inline void ct_step(ct_state *s, double u) {
  double ue = W0 * u + W1 * s->um1 + W2 * s->um2;
  const double h = 1.0 / NSUB;
  double x1 = s->x1, x2 = s->x2, a1, a2, b1, b2, c1, c2_, d1, d2;
  for (int i = 0; i < NSUB; i++) {
    rhs(x1, x2, ue, &a1, &a2);
    rhs(x1 + 0.5*h*a1, x2 + 0.5*h*a2, ue, &b1, &b2);
    rhs(x1 + 0.5*h*b1, x2 + 0.5*h*b2, ue, &c1, &c2_);
    rhs(x1 + h*c1, x2 + h*c2_, ue, &d1, &d2);
    x1 += (h/6.0) * (a1 + 2*b1 + 2*c1 + d1);
    x2 += (h/6.0) * (a2 + 2*b2 + 2*c2_ + d2);
    clampx(&x1, &x2);
  }
  s->x1 = x1; s->x2 = x2; s->um2 = s->um1; s->um1 = u;
}
static inline double ct_out(const ct_state *s) {
  double y = A + s->x2 + C2 * s->x2 * s->x2; return y < YCLIP ? y : YCLIP;
}
#ifdef CT_MAIN
int main(int argc, char **argv) {
  // usage: ct_main  -> reads N, x1, x2, u[-2], u[-1] then N inputs from stdin, writes N outputs; or --bench
  int n; double x1, x2, um2, um1;
  if (argc > 1 && argv[1][0]=='-') {
    int reps = 20000; n = 1024; double acc = 0;
    ct_state s = { 3.739683641629819, 3.3841867891864097, 3.0, 3.0 };
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < reps; r++) {
      s.x1 = 3.739683641629819; s.x2 = 3.3841867891864097;
      for (int t = 0; t < n; t++) { double u = 3.0 + 2.0 * sin(0.013 * t + r); ct_step(&s, u); acc += ct_out(&s); }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / ((double)reps * n);
    printf("%.3f ns_per_step (checksum %.6f)\n", ns, acc);
    return 0;
  }
  if (scanf("%d %lf %lf %lf %lf", &n, &x1, &x2, &um2, &um1) != 5) return 1;
  ct_state s = { x1, x2, um1, um2 };
  for (int t = 0; t < n; t++) { double u; if (scanf("%lf", &u) != 1) return 2; printf("%.12f\n", ct_out(&s)); ct_step(&s, u); }
  return 0;
}
#endif
