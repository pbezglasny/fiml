# ADR 0003: Define sample-SMA partial-window semantics

Status: accepted  
Date: 2026-07-25

## Context

The sample-count `SimpleMovingAverage` maintained the correct partial sum while
a window filled but always divided that sum by the configured period. A
10-period SMA receiving its first value of 100 therefore emitted 10 instead of
100. The result was neither a moving average nor an explicit missing value, and
introduced an artificial ramp into ML features.

The broader feature warm-up contract remains unsettled. In particular, whether
all rolling indicators should withhold output until a complete window is ready
is tracked separately from correcting the sample-SMA calculation.

The library is still in development, and its feature-set JSON version currently
identifies the schema rather than every numerical-behavior revision.

## Decision

After at least one matching sample, a sample-window SMA emits the arithmetic
mean of the samples currently available to that window. Its divisor is:

```text
min(number of buffered samples, configured window period)
```

Once the period is full, the SMA continues to average exactly the most recent
`period` samples.

The standalone indicator's no-sample behavior remains unchanged. Compiled
extractor cells remain NaN until their feature first writes, after which a
sample-window SMA writes its partial mean.

This decision does not define a uniform readiness policy for other indicators
and does not require outputs to remain NaN until a full window is available.
That policy remains deferred.

`FEATURE_SET_FORMAT_VERSION` remains `1.0.0`. The pre-release correctness fix
does not introduce artifact-version churn.

## Consequences

- Partial sample-window SMA values are on the same scale as converged values.
- A constant input produces the same constant SMA from its first sample.
- Multi-window SMA outputs use an independent divisor capped by each period.
- Existing pre-release streams and saved model features that included partial
  SMA windows will produce different values when recomputed.
- A future uniform warm-up policy may withhold these correct partial means from
  feature outputs without changing the underlying SMA arithmetic.
