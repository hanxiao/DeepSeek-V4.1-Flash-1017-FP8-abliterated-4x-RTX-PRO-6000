import numpy as np
from q2_layer import make_problem, quant
from q2_ops import rel_rms
W4, s0, X, H = make_problem(act_sigma=0.6, spikes=8)
c_cur = np.array([-1.5,-0.5,0.5,1.5]); c_r4 = np.array([-4.,-1.,1.,4.])
for name,c in (("current",c_cur),("ratio4",c_r4)):
    q = quant(W4, s0, c, None, mode="ls")
    print(f"  {name:8s} + LS scale, plain: {rel_rms(W4,q)[1]:.2f} % weight RMS")
