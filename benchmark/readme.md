## Benchmarking

I have a benchmarking suite in `benchmark.py`. As of this current commit (June 9th, 2025), the time taken is **0.19s** averaged across 100 runs with stdev 0.01. This is after doing a bunch of rote speed optimizations so it's probably the quickest I'd get for a while.

I want to have this here so that I can ensure that the time does not massively balloon with future modifications.

## Profiling

`benchmark-profiler.py` is what I used to optimize speed. Run `python benchmark-profiler.py > cprofile.txt` to get an in depth profile of functions used and how long they take. This is great for finding anomalies.

The output would look something like this:

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
43212/630    0.048    0.000    0.824    0.001 expr.py:99(wrapper)
       66    0.000    0.000    0.677    0.010 integration.py:55(integrate)
       66    0.001    0.000    0.644    0.010 integration.py:232(integrate)
       42    0.000    0.000    0.552    0.013 test_utils.py:10(assert_integral)
9242/8586    0.015    0.000    0.375    0.000 expr.py:1144(__new__)
      142    0.000    0.000    0.374    0.003 integration.py:147(_cycle)

```

## Logging

`benchmark-profiler.py` also keeps track of the time it takes for each integral to run and creates a plot of it. See `integration_log.png` for the bar chart; see `integration_log.txt` for some more detailed numerical breakdowns.
