#pragma once

struct vanillagrad
{
    void outputgrad(const float *a1, const float *a2, const float *y, unsigned n, unsigned m, float *delta, float *d2);
    void hiddengrad(const float *a1, const float *a0, const float *d2, const float *w, int n, int m, int p, float *delta, float *d1);
};
