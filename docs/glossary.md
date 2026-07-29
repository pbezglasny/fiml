# Glossary

## Feature set

The complete, versioned definition used to build an extractor. It is the parity
artifact shared by Python training and Rust serving.

## Feature group

A serialization concept that groups indicators by scope. A group is either
global or belongs to one normalized market symbol. Runtime definitions remain
flat.

## Indicator definition

An indicator name, its constructor options, and the scope inherited from its
feature group. One indicator definition may own several adjacent feature cells.

## Indicator identity

The fields that distinguish runtime indicator instances independently of their
windows. Identity includes scope, indicator kind, and applicable fields such as
value source, aggregation, or UTC offset.

## Feature cell

One numeric model input in the extractor's feature vector. For a multi-window
indicator, each configured window owns one adjacent feature cell.

## Canonical order

The deterministic feature-vector order computed from global scope first,
normalized symbol, indicator name, indicator identity, and finally authored
window order.

## Serialization module

The private module that converts between the hierarchical JSON contract and the
flat runtime feature definitions. Its interface is `FeatureSet` serialization
and deserialization.

## Structural validation

Validation required to convert the JSON contract without ambiguity, including
field shape, version, scalar syntax, feature-group uniqueness, and scope.

## Semantic validation

Validation performed during extractor compilation, including window rules,
indicator identity uniqueness, output capacity, and generated feature-name
uniqueness.
