Minimization problems
~~~~~~~~~~~~~~~~~~~~~

Solver includes 3 different minimization problems.

**Classical form**

.. math:: \min_{\Theta} \sum_{\alpha=1}^{q} \left[\| \bar{S} \bar{u}^\alpha - f^\alpha \|_{i} + \lambda \|\bar{b} \bar{u}^\alpha - g^\alpha\|_{j} \right]_{X}

**Weak form**
.. math:: \min_{\Theta} \sum_{\alpha=1}^{q} \int_{X} \left[\bar{S}(\bar{u}^\alpha) - f^\alpha\right] \cdot \phi^\alpha \, \mathrm{d}X + \left[\lambda \|\bar{b} \bar{u}^\alpha - g^\alpha\|_{j} \right]_{X}

**Causal form**
.. math:: \min_{\Theta} \sum_{\alpha=1}^{q} \left[\omega * \| \bar{S} \bar{u}^\alpha - f^\alpha \|_{i} + \lambda \|\bar{b} \bar{u}^\alpha - g^\alpha\|_{j} \right]_{X}
where
.. math:: \omega = {\omega_0 ... \omega_i} 
.. math:: \omega_i = \exp(-\epsilon \sum_{\k=1}^{i-1}\bar{S} \bar{u}^\alpha - f^\alpha), for i = 2,3,...N_t, N_t is number of grid points, \epsilon is constant.

