# Project 1: Physics-Informed Groundwater Modeling (PINN)

## Overview
This project implements a Physics-Informed Neural Network (PINN) to solve the inverse groundwater problem in the Souss-Massa basin. The goal is to estimate the spatial distribution of Transmissivity ($T$) using sparse piezometric head ($h$) observations by enforcing the 2D Boussinesq equation as a physics loss function.

## Mathematical Framework
The governing equation for transient flow in an unconfined aquifer is the Boussinesq equation:

$$S \frac{\partial h}{\partial t} = \frac{\partial}{\partial x} \left( T \frac{\partial h}{\partial x} \right) + \frac{\partial}{\partial y} \left( T \frac{\partial h}{\partial y} \right) + R - P$$

Where:
- $h(x,y,t)$: Hydraulic head [L]
- $T(x,y)$: Transmissivity [L²/T] (Parameter to be learned)
- $S$: Storage coefficient [dimensionless]
- $R$: Recharge (precipitation/irrigation return) [L/T]
- $P$: Pumping [L/T]

## Methodology
1.  **Network Architecture**: A fully connected neural network takes $(x, y, t)$ as input and outputs $\hat{h}$ and $\hat{T}$.
2.  **Loss Function**:
    -   $L_{total} = L_{data} + \lambda L_{physics}$
    -   $L_{data} = MSE(h_{pred} - h_{obs})$ (at sparse observation points)
    -   $L_{physics} = MSE(\text{Residual of Boussinesq Eq})$ (at collocation points throughout the domain)
3.  **Data Sources**:
    -   Piezometric maps from Souss-Massa ABH reports (digitized).
    -   Recharge estimates from CHIRPS/remote sensing.

## Usage
1.  Install requirements: `pip install -r requirements.txt`
2.  Define domain geometry and boundary conditions in `model_architecture.py`.
3.  Load sparse measurement data.
4.  Train the model to minimize $L_{total}$.
5.  Extract the learned $T(x,y)$ field.

