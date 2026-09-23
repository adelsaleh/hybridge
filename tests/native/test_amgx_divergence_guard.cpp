// Standalone unit check of the real native guard. Compile/run only on request.
#include <convergence/residual_divergence.h>
#include <cassert>
#include <limits>
#include <initializer_list>

using amgx::ResidualDivergenceGuard;

int main()
{
    const double eps = std::numeric_limits<double>::epsilon();
    // Large initial magnitude and temporary growth must permit recovery.
    ResidualDivergenceGuard transient(1000, 5, 10, 1e9, eps);
    assert(!transient.observe(1, 1));
    for (int i = 2; i <= 10; ++i) assert(!transient.observe(1e6, i));
    for (int i = 11; i <= 14; ++i) assert(!transient.observe(2000, i));
    assert(!transient.observe(0.1, 15));
    for (int i = 16; i <= 19; ++i) assert(!transient.observe(200, i));
    assert(transient.observe(200, 20));
    assert(transient.excessive(200));
    assert(!transient.excessive(0.01));
    transient.unconfirmed(); // A good explicit b-A*x cancels the suspicion.
    for (int i = 21; i <= 24; ++i) assert(!transient.observe(200, i));
    assert(transient.observe(200, 25));

    // A relative threshold is invariant when the whole residual history scales.
    for (double scale : {1e-80, 1.0, 1e80})
    {
        ResidualDivergenceGuard guard(1000, 5, 0, 1e5 * scale, eps);
        assert(!guard.observe(scale, 1));
        for (int i = 2; i <= 5; ++i) assert(!guard.observe(2000 * scale, i));
        assert(guard.observe(2000 * scale, 6));
    }
    // A tiny recursive minimum must not turn FP32 roundoff into divergence.
    ResidualDivergenceGuard floor(1000, 5, 0, 1e5, std::numeric_limits<float>::epsilon());
    assert(!floor.observe(1e-20, 1));
    for (int i = 2; i <= 20; ++i) assert(!floor.observe(0.1, i));
    for (int i = 21; i <= 24; ++i) assert(!floor.observe(1e4, i));
    assert(floor.observe(1e4, 25));

    for (double bad : {std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()})
    {
        ResidualDivergenceGuard guard(1000, 5, 10, 1, eps);
        assert(guard.observe(bad, 1)); // Non-finite bypasses grace and patience.
        ResidualDivergenceGuard disabled(-1, 5, 10, 1, eps);
        assert(!disabled.observe(bad, 1));
    }
    // A new solve starts with fresh history, even when solver objects are reused.
    ResidualDivergenceGuard fresh(1000, 5, 10, 1e5, eps);
    assert(!fresh.observe(1e5, 1));
}
