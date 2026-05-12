# CUDA / GPU Kernel Learning

A 6-month structured journey from RTL engineering into GPU kernel programming and ML systems engineering. Every kernel is profiled with Nsight Compute from Day 1; the profiler is ground truth.

## Layout

- `month_01/` — CUDA programming model, first kernels, profiling discipline
- `month_02/` — Matmul optimization gauntlet (Simon Boehm steps)
- `month_03/` — Triton DSL + transformer compute graph context
- `month_04/` — Transformer kernels from scratch (softmax, LayerNorm, attention)
- `month_05/` — FlashAttention from scratch
- `month_06/` — Writeups, portfolio polish
- `notes/` — Reading notes, bug log
- `profiles/` — Saved `.ncu-rep` / `.nsys-rep` files

## Build convention

Each week directory has a `build/` subdirectory (gitignored) where compiled binaries land. Sources stay clean.

## Hardware

Developed on an NVIDIA RTX A5000 (Ampere, compute capability 8.6, 24 GB GDDR6).
See `month_01/week_01/device_query_output.txt` for the full device spec captured on Day 1.

## Status

Month 1, Week 1 — in progress.
