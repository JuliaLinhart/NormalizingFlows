# NormalizingFlows
Pytorch implementation of some flow architectures.

The core structure of the flow is implemented in `flows.py`. 

The `Flow` object is defined by a *base distribution* $p_u$ and a *transformation* $T$ with parameters $\Phi$.
- We can learn the parameters $\Phi$, fitting the flow-based model to a set of observations `x_samples` by MLE in order to approximate the corresponding density $p_x$ by calling the `train` method.
- We can `sample` from this `Flow`-object and evaluate its density function `learned_pdf` based on the following formulas:
$$ x = T(u; \Phi), \quad u \sim p_u(u)$$
$$
\begin{aligned}
p_{\mathrm{x}}(x; \Phi) & = p_{\mathrm{u}}(u)\left|\operatorname{det} J_{T}(u; \Phi)\right|^{-1} \quad \text { where } \quad u=T^{-1}(x; \Phi) \\
& = p_{\mathrm{u}}\left(T^{-1}(x; \Phi)\right)\left|\operatorname{det} J_{T^{-1}}(x; \Phi)\right|
\end{aligned}
$$

The different transformations are implemented in `transforms.py`:
- `AffineElementwiseTransform` : $T : \mathbf{u} = (u_1, \dots, u_d) \mapsto 
T(\mathbf{u}) = (a_1u_1 + b_1, \dots, a_du_d + b_d) = (x_1, \dots, x_d)$. 
The parameters to learn are $\Phi = (\mathbf{a},\mathbf{b})$
- `PositiveLinearTransformation` : $T: \mathbf{u} = (u_1, \dots, u_d) \mapsto T(u)=\mathbf{A}u + \mathbf{b} = {x}$, 
where $\mathbf{A}$ is a matrix with non-negative components only. 
The parameters to learn are $\Phi = (\mathbf{A},\mathbf{b})$.

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}) 
