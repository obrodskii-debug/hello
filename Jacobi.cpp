#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;

struct Result {
    int    iters;      // выполненное число итераций
    double diff;       // последняя норма max|x_new - x_old|
};

Result jacobi(const vector<double>& A, const vector<double>& b,
    vector<double>& x, int N, double eps, int max_iter)
{
    vector<double> x_new(N, 0.0);
    double diff = 0.0;
    int iter = 0;

    do {
        diff = 0.0;

#pragma omp parallel
        {
            double local_diff = 0.0;          // частный максимум потока

#pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                double s = 0.0;
                const double* row = &A[i * N];
                for (int j = 0; j < N; j++)
                    s += row[j] * x[j];
                s -= row[i] * x[i];           
                x_new[i] = (b[i] - s) / row[i];
                local_diff = max(local_diff, fabs(x_new[i] - x[i]));
            }
            // неявный барьер в конце omp for

            // Объединение частных максимумов, как в ручной редукции
#pragma omp critical
            diff = max(diff, local_diff);
        }

        x.swap(x_new);                        // новое приближение становится текущим
        iter++;
    } while (diff > eps && iter < max_iter);

    return { iter, diff };
}

int main(int argc, char** argv)
{
     int    N = (argc > 1 ? atoi(argv[1]) : 2000);
     double eps = (argc > 2 ? atof(argv[2]) : 1e-10);
     int    max_iter = (argc > 3 ? atoi(argv[3]) : 10000);

#ifdef _OPENMP
    cout << "OpenMP ON, max threads = " << omp_get_max_threads()
        << ", procs = " << omp_get_num_procs() << endl;
#else
    cout << "OpenMP OFF" << endl;
#endif

    vector<double> A(N * N), b(N), x_true(N), x(N, 0.0);
    srand(12345);
    for (int i = 0; i < N; i++) {
        double off = 0.0;
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            double v = (double)rand() / RAND_MAX;      // [0, 1]
            A[(size_t)i * N + j] = v;
            off += fabs(v);
        }
        A[(size_t)i * N + i] = 2.0 * off + 1.0;       // диагональное преобладание
        x_true[i] = i + 1;
    }
    for (int i = 0; i < N; i++) {
        double s = 0.0;
        for (int j = 0; j < N; j++)
            s += A[(size_t)i * N + j] * x_true[j];
        b[i] = s;
    }

    auto t0 = chrono::steady_clock::now();
    Result r = jacobi(A, b, x, N, eps, max_iter);
    double dt = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

    // Проверка: погрешность относительно точного решения
    double err = 0.0;
    for (int i = 0; i < N; i++)
        err = max(err, fabs(x[i] - x_true[i]));

    cout << "N = " << N << ", eps = " << eps << endl;
    cout << "Iterations = " << r.iters << ", last diff = " << r.diff << endl;
    cout << "Max error vs exact solution = " << err << endl;
    cout << "Time = " << dt << " s" << endl;
    return 0;
}