# ADR 0003: Keep wind coefficients fixed after tree edits

- Status: Accepted for first release
- Date: 2026-08-31

## Context

The current workflow consumes precomputed wind-coefficient rasters. Recomputing local airflow after every tree placement would require a separate aerodynamic model or CFD pipeline and would expand both influence range and computation cost.

## Decision

The first incremental release updates radiation-driven shade, vegetation sky-view effects, Tmrt, and UTCI while retaining the baseline wind coefficients. The UI and result metadata disclose this limitation.

## Consequences

- Tree shade effects can be explored interactively.
- Results do not represent tree-induced changes to local wind speed, humidity, or evapotranspiration.
- Claims must use language such as radiation-driven thermal comfort or tree-shade design support.

## Revisit when

A validated fast wind-response model and corresponding full-domain oracle are available.
