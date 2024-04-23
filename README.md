# Mixture Density Networks in Pytorch

Pytorch implementation of mixture density networks given by [1] using the loss function 
```math
E(\textbf{w}) = -\sum_{n=1}^{N}\ln \left\{\sum_{k=1}^{K}\pi_k(\textbf{x}_n,\textbf{w})\mathcal{N}(\textbf{t}_n|\mu_{k}(\textbf{x}_n,\textbf{w}), \sigma_k^2 (\textbf{x}_n,\textbf{w})) \right\}
```

The corresponding blog post can be found [here](https://www.brianjsl.com/blog/2024/04/22/mixture_density_networks/).

## References

[1] Bishop, C. M. Mixture density networks. (1994).
