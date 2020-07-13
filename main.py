# initial imports from qiskit

from qiskit                                        import *
from qiskit.chemistry.drivers                      import UnitsType,HFMethodType
from qiskit.chemistry.core                         import Hamiltonian,TransformationType,QubitMappingType
from qiskit.chemistry.components.initial_states    import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.aqua.components.optimizers             import L_BFGS_B,COBYLA
from qiskit.aqua.algorithms                        import VQE
from qiskit.aqua                                   import QuantumInstance,aqua_globals

# very verbose output, helpful for debugging
#import logging
#from   qiskit.chemistry import set_qiskit_chemistry_logging
#set_qiskit_chemistry_logging(logging.DEBUG)

# generation of a molecule object; 
# PySCF driver slightly tweaked to allow molecular symmetries to be enforced at SCF level
# and visualization of SCF orbitals
import sys
sys.path.append('./pyscfd')
from pyscfdriver import *
from utils       import *

outfile   = open('qiskit_vqe.txt','w')
driver    = PySCFDriver(atom='''S 0.0000 0.0000 0.1030; H 0.0000 0.9616 -0.8239; H 0.0000 -0.9616 -0.8239''',
                        unit=UnitsType.ANGSTROM,charge=0,spin=0,basis='sto-6g',hf_method=HFMethodType.RHF,symgroup='C2v',outfile=outfile)
molecule  = driver.run()

# qubit representation of the Hamiltonian (possibly in an active space of chosen orbitals)
orb_red   = [0,1,2,3,4]     # simple frozen core
orb_red   = [0,1,2,3,4,8]   # simple frozen core and out-of-plane orbital reduction
core      = Hamiltonian(transformation=TransformationType.FULL,qubit_mapping=QubitMappingType.PARITY,
                        two_qubit_reduction=True,freeze_core=False,orbital_reduction=orb_red)
H_op,A_op = core.run(molecule)

# tapering off symmetries to reduce the number of qubits
z2syms,sqlist           = None,None
H_op,A_op,z2syms,sqlist = taper(molecule,core,H_op,A_op,outfile)

# hartree-fock and VQE-qUCCSD
init_state = HartreeFock(num_orbitals=core._molecule_info['num_orbitals'],qubit_mapping=core._qubit_mapping,
                         two_qubit_reduction=core._two_qubit_reduction,num_particles=core._molecule_info['num_particles'],sq_list=sqlist)
circuit    = init_state.construct_circuit()

outfile.write("\nHartree-Fock energy %f \n" % (molecule.hf_energy))
outfile.write("\nHartree-Fock circuit\n")
outfile.write(str(circuit.draw())+"\n")

var_form  = UCCSD(num_orbitals=core._molecule_info['num_orbitals'],num_particles=core._molecule_info['num_particles'],active_occupied=None,active_unoccupied=None,
                  initial_state=init_state,qubit_mapping=core._qubit_mapping,two_qubit_reduction=core._two_qubit_reduction,num_time_slices=1,z2_symmetries=z2syms)

optimizer = L_BFGS_B(maxiter=1000)
algo      = VQE(H_op,var_form,optimizer,aux_operators=A_op)

backend          = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend=backend)
algo_result      = algo.run(quantum_instance)
get_results(H_op,A_op,molecule,core,algo_result,outfile)
print_UCCSD_parameters(molecule,core,var_form,algo_result,z2syms,sqlist,orb_red,outfile)

