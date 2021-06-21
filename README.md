# Child Mortality

Ratio metrics over every dimensional cut of a table.

## Data model

A table has a unique row id, many categorical dimension columns (each with only a few
distinct values compared with the number of rows), and two numeric measure columns. For any
group of rows the ratio metric is

    sum(measure1) / sum(measure2)

## Problem 1: full cube

For the first k dimensions (k = 3 in the example), compute the ratio for every 1-, 2-, ...,
k-dimensional cut: every subset of those dimensions, and every combination of values within
each subset.

## Problem 2: thresholded cube

With many dimensions (say 50) the number of cuts explodes. Compute only the cells that contain
at least a minimum number of rows, and skip work for cells that cannot reach it: a finer cut
never has more rows than any coarser cut that contains it, and a threshold larger than the
table needs no computation at all.

## Application: under-5 mortality

Where birth and death registries are incomplete, household surveys of women give a proxy for
under-5 mortality, CD/CEB = children died / children ever born, usually split by the mother's
5-year age group (younger mothers give a more recent estimate).

`women.csv` holds one row per woman and `household.csv` one row per household (DHS model
dataset). A woman's household is found by `Cluster number` + `Household number`.

    CD/CEB = sum(Sons who have died + Daughters who have died) / sum(Total children ever born)

Dimensions:

- `Age in 5-year groups`
- `Type of place of residence`
- `Number of household members`
- `Source of drinking water`
- `Time to get to water source (minutes)`
- `Type of toilet facility`
- `Has electricity`
- `Has radio`
- `Age of head of household`

The code should generalize to any dataset in the format above.
