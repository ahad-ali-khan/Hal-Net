# HAL-Net

A lightweight CNN that estimates analog film halation parameters from highlight regions and synthesizes a matching warm glow.

Trained on 100 handpicked CineStill 800T film scans from Flickr and Lomography. Inspired by FGA-NN (Ameur et al., arXiv 2506.14350, 2025), which models film grain but does not address halation.

## What it does

Upload any image. HAL-Net analyzes the highlight regions, estimates three parameters (radius, intensity, warmth), and synthesizes a physically motivated halation effect. Manual override sliders let you adjust the predicted parameters and see the result in real time.

## Parameters

- **Radius:** how far the glow spreads from the highlight edge (maps to a Gaussian sigma of 0-32px at 256px, scaled proportionally at higher resolutions)
- **Intensity:** strength of the glow relative to the surrounding area
- **Warmth:** color bias from neutral white (0.0) to CineStill red-orange (1.0)

## Evaluation

Per-parameter mean absolute error on the 15-image held-out test split:

| Method             | Radius MAE | Intensity MAE | Warmth MAE | Mean MAE |
|--------------------|------------|---------------|------------|----------|
| Mean-pred baseline | 0.012      | 0.317         | 0.328      | 0.219    |
| HAL-Net            | 0.015      | 0.247         | 0.218      | 0.160    |

HAL-Net beats the mean-prediction baseline on intensity and warmth, the two parameters that drive the visual appearance of the glow. Radius MAE is near-zero for both because the pseudo-label distribution is narrow (most images land around 0.09). The main weakness is intensity under-prediction on the strongest examples, traced to label sparsity at the high end rather than an architecture problem.

## What is halation

Halation is the red-orange glow around bright light sources in analog film. It occurs when light passes through the emulsion, reflects off the film base, and re-exposes the silver halide crystals from behind. CineStill 800T is the most well-known example: its anti-halation backing was removed during processing for cinema use, making the effect visible and distinctive.

## Dataset

100 CineStill 800T scans from Flickr and Lomography, selected for visible halation around practical light sources (street lamps, neon signs, candles, windows). Split 70/15/15 train/val/test.

## Architecture

3 strided conv layers (32, 64, 128 channels) + 2 residual blocks + adaptive average pool + 2-layer regression head with Sigmoid output. 693k parameters. Trains in under 5 minutes on a T4 GPU.

## Based on

FGA-NN: Film Grain Analysis Using Neural Networks  
Ameur et al., arXiv 2506.14350, 2025  
https://arxiv.org/abs/2506.14350

## Live demo

https://huggingface.co/spaces/ahadstfu/hal-net  
![HAL-Net demo](./demo.jpg)