// Standalone CPU test; compile with -I../../../AMGX/include from this directory.
#include <solvers/natural_ilu_pattern.h>
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
using namespace amgx::multicolor_ilu_solver;

// Independent dense, pivot-major symbolic Gaussian elimination oracle.
static std::vector<int> dense_levels(int n, const std::vector<int> &r,
                                     const std::vector<int> &c, int fill)
{
    const int inf = 1000000;
    std::vector<int> levels(n*n, inf);
    for (int i=0; i<n; ++i)
        for (int p=r[i]; p<r[i+1]; ++p) levels[i*n+c[p]]=0;
    for (int j=0; j<n; ++j)
        for (int i=j+1; i<n; ++i)
            if (levels[i*n+j] <= fill)
                for (int k=j+1; k<n; ++k)
                {
                    const int level=levels[i*n+j]+levels[j*n+k]+1;
                    if (level <= fill) levels[i*n+k]=std::min(levels[i*n+k],level);
                }
    return levels;
}

int main()
{
    std::mt19937 rng(20261031);
    int cases=0, level_two_entries=0;
    for (int trial=0; trial<120; ++trial)
    {
        const int n=12+trial%21;
        std::vector<int> r(1,0),c;
        for (int i=0; i<n; ++i)
        {
            for (int j=0; j<n; ++j)
                if (i==j || rng()%100<17) c.push_back(j);
            r.push_back(c.size());
        }
        for (int fill : {0,1,2,3,32})
        {
            auto actual=natural_ilu_pattern(n,r.data(),c.data(),fill);
            auto expected=dense_levels(n,r,c,fill);
            std::vector<int> er(1,0),ec,el;
            for (int i=0; i<n; ++i)
            {
                for (int j=0; j<n; ++j)
                    if (expected[i*n+j]<=fill)
                    {
                        ec.push_back(j); el.push_back(expected[i*n+j]);
                    }
                er.push_back(ec.size());
                for (int p=r[i]; p<r[i+1]; ++p)
                {
                    int q=actual.source_to_factor[p];
                    assert(q>=actual.rows[i] && q<actual.rows[i+1]);
                    assert(actual.columns[q]==c[p] && actual.levels[q]==0);
                }
            }
            assert(actual.rows==er && actual.columns==ec && actual.levels==el);
            if (fill==0) assert(actual.rows==r && actual.columns==c);
            if (fill==2) level_two_entries+=std::count(el.begin(),el.end(),2);
            ++cases;
        }
    }
    assert(level_two_entries>0);
    const std::vector<int> r{0,2,4,6}, c{0,2,0,1,1,2};
    auto pattern=natural_ilu_pattern(3,r.data(),c.data(),1);
    assert(pattern.columns.size()>c.size());
    int rejected=0;
    try { natural_ilu_pattern(3,r.data(),c.data(),1,c.size()); }
    catch(const std::length_error &) { ++rejected; }
    auto bad=c; std::swap(bad[0],bad[1]);
    try { natural_ilu_pattern(3,r.data(),bad.data(),1); }
    catch(const std::invalid_argument &) { ++rejected; }
    bad=c; bad[1]=0;
    try { natural_ilu_pattern(3,r.data(),bad.data(),1); }
    catch(const std::invalid_argument &) { ++rejected; }
    try { natural_ilu_pattern(3,r.data(),c.data(),-1); }
    catch(const std::invalid_argument &) { ++rejected; }
    try { natural_ilu_pattern(3,r.data(),c.data(),std::numeric_limits<int>::max()); }
    catch(const std::invalid_argument &) { ++rejected; }
    assert(rejected==5);
    std::cout << "PASS: " << cases << " independently checked symbolic patterns; "
              << rejected << " input/capacity rejections; " << level_two_entries
              << " level-two entries exercised\n";
}
