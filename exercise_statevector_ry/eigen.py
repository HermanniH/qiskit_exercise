from qiskit                 import *
from qiskit.aqua.algorithms import NumPyEigensolver
from prettytable            import PrettyTable

import numpy as np

def eigen_decomposition(molecule,H_op,dE,A_op,dA,outf):
    ee = NumPyEigensolver(operator=H_op,k=2**H_op.num_qubits,aux_operators=A_op)
    ee = ee.run()
    t = PrettyTable(['Energy','N','Sz','S^2','Dx','Dy','Dz'])
    for x,v in zip(ee['eigenvalues'],ee['aux_operator_eigenvalues']):
        x,v = np.real(x+dE),[np.real(vi[0]+dAi) for vi,dAi in zip(v,dA)]
        if(x<molecule.hf_energy):
           t.add_row([str(round(x,6))]+[str(round(w,6)) for w in v])
    outf.write(str(t))
    outf.write("\n")

