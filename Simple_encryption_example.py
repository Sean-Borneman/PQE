from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

def create_lwe_quantum_circuit(add_error=True, error_strength=0.1):
    """
    Create a quantum circuit for LWE problem demonstration with 2x2 matrix A
    
    Args:
        add_error: Whether to add error terms to b
        error_strength: How strong the error should be (0.1 = small, 0.5 = large)
    
    Registers:
    - s_reg: Secret key register (2 qubits for 2D vector)
    - a_reg: Public matrix A register (4 qubits for 2x2 matrix)  
    - b_reg: Public result b register (2 qubits for 2D result vector)
    - e_reg: Error register (2 qubits for 2D error vector)
    - aux_reg: Auxiliary qubits for matrix multiplication
    """
    
    # Create quantum registers
    s_reg = QuantumRegister(2, 's')  # Secret key vector [s0, s1]
    a_reg = QuantumRegister(4, 'a')  # 2x2 Matrix A: [a00, a01, a10, a11]
    b_reg = QuantumRegister(2, 'b')  # Result vector [b0, b1]
    e_reg = QuantumRegister(2, 'e')  # Error vector [e0, e1]
    aux_reg = QuantumRegister(2, 'aux')  # Auxiliary qubits for computation
    u_reg = QuantumRegister(2, 'u')  # Ciphertext u vector [u0, u1]
    v_reg = QuantumRegister(2, 'v')  # Ciphertext v vector [v0, v1]
    
    # Classical registers for measurement
    s_classical = ClassicalRegister(2, 'sc')
    a_classical = ClassicalRegister(4, 'ac')
    b_classical = ClassicalRegister(2, 'bc')
    e_classical = ClassicalRegister(2, 'ec')
    u_classical = ClassicalRegister(2, 'uc')
    v_classical = ClassicalRegister(2, 'vc')
    
    # Create the circuit
    qc = QuantumCircuit(s_reg, a_reg, b_reg, e_reg, aux_reg, u_reg, v_reg,
                       s_classical, a_classical, b_classical, e_classical, u_classical, v_classical)
    
    # Initialize the LWE parameters
    # Set secret key s = [1, 0] for demo
    qc.x(s_reg[0])  # s0 = 1
    # s_reg[1] remains 0
    
    # Set 2x2 matrix A = [[1,1], [0,1]] for demo
    # a_reg[0] = a00, a_reg[1] = a01, a_reg[2] = a10, a_reg[3] = a11
    qc.x(a_reg[0])  # a00 = 1
    qc.x(a_reg[1])  # a01 = 1  
    # a_reg[2] remains 0  # a10 = 0
    qc.x(a_reg[3])  # a11 = 1
    
    # Create small error vector (much smaller than before)
    if add_error:
        # Instead of Hadamard (50% error), use small rotations for realistic error
        qc.ry(error_strength * np.pi, e_reg[0])  # Small probability of e0=1
        qc.ry(error_strength * np.pi, e_reg[1])  # Small probability of e1=1
    
    qc.barrier(label="Setup complete")
    
    # Calculate b = A*s + e (mod 2)
    # Matrix multiplication: b = A*s
    # b0 = a00*s0 XOR a01*s1 = 1*1 XOR 1*0 = 1 XOR 0 = 1
    # b1 = a10*s0 XOR a11*s1 = 0*1 XOR 1*0 = 0 XOR 0 = 0
    # Expected result: b = [1, 0]
    
    # More explicit matrix multiplication using auxiliary qubits
    # Compute a00*s0 in aux[0]
    qc.ccx(a_reg[0], s_reg[0], aux_reg[0])
    # Compute a01*s1 in aux[1] 
    qc.ccx(a_reg[1], s_reg[1], aux_reg[1])
    # b0 = aux[0] XOR aux[1] = a00*s0 XOR a01*s1
    qc.cx(aux_reg[0], b_reg[0])
    qc.cx(aux_reg[1], b_reg[0])
    
    # Reset auxiliary qubits
    qc.ccx(a_reg[0], s_reg[0], aux_reg[0])  # Reset aux[0]
    qc.ccx(a_reg[1], s_reg[1], aux_reg[1])  # Reset aux[1]
    
    # Compute a10*s0 in aux[0]
    qc.ccx(a_reg[2], s_reg[0], aux_reg[0])
    # Compute a11*s1 in aux[1]
    qc.ccx(a_reg[3], s_reg[1], aux_reg[1])
    # b1 = aux[0] XOR aux[1] = a10*s0 XOR a11*s1
    qc.cx(aux_reg[0], b_reg[1])
    qc.cx(aux_reg[1], b_reg[1])
    
    # Reset auxiliary qubits
    qc.ccx(a_reg[2], s_reg[0], aux_reg[0])  # Reset aux[0]
    qc.ccx(a_reg[3], s_reg[1], aux_reg[1])  # Reset aux[1]
    
    qc.barrier(label="Matrix mult done")
    
    # Add error: b = b + e (only if add_error is True)
    if add_error:
        qc.cx(e_reg[0], b_reg[0])  # b0 += e0
        qc.cx(e_reg[1], b_reg[1])  # b1 += e1
        qc.barrier(label="Error added")
    
    # LWE Encryption simulation (simplified)
    # For message m=[0,0], compute:
    # u = A^T * r (assuming r=[1,0] for demo)
    # v = b^T * r + m (assuming m=[0,0])
    
    # Compute u = A^T * r where r = [1,0]
    # u0 = a00*r0 + a10*r1 = a00*1 + a10*0 = a00 = 1
    # u1 = a01*r0 + a11*r1 = a01*1 + a11*0 = a01 = 1
    qc.cx(a_reg[0], u_reg[0])  # u0 = a00
    qc.cx(a_reg[1], u_reg[1])  # u1 = a01
    
    # Compute v = b^T * r + m where r = [1,0], m = [0,0]  
    # v0 = b0*r0 + b1*r1 + m0 = b0*1 + b1*0 + 0 = b0
    # v1 = b1 (simplified for demo)
    qc.cx(b_reg[0], v_reg[0])  # v0 = b0
    qc.cx(b_reg[1], v_reg[1])  # v1 = b1
    
    qc.barrier(label="Encryption done")
    
    # Measure all registers
    qc.measure(s_reg, s_classical)
    qc.measure(a_reg, a_classical)
    qc.measure(b_reg, b_classical)
    qc.measure(e_reg, e_classical)
    qc.measure(u_reg, u_classical)
    qc.measure(v_reg, v_classical)
    
    return qc

def simulate_lwe_circuit(add_error=True, error_strength=0.1, shots=1024):
    """
    Simulate the LWE quantum circuit and return results
    """
    # Create the circuit
    qc = create_lwe_quantum_circuit(add_error=add_error, error_strength=error_strength)
    
    # Use Aer simulator
    simulator = AerSimulator()
    
    # Transpile the circuit for the simulator
    transpiled_qc = transpile(qc, simulator)
    
    # Run the simulation
    job = simulator.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    return qc, counts

def analyze_lwe_results(counts, add_error=True):
    """
    Analyze the results of the LWE circuit simulation with 2x2 matrix
    """
    error_status = "with error" if add_error else "without error"
    print(f"LWE Quantum Circuit Results Analysis (2x2 Matrix, {error_status})")
    print("=" * 60)
    
    total_shots = sum(counts.values())
    
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
        # Parse the measurement results
        # Format: vc[1:0] uc[1:0] ec[1:0] bc[1:0] ac[3:0] sc[1:0]
        if ' ' in bitstring:
            parts = bitstring.split()
        else:
            # Parse as continuous string: vc(2) uc(2) ec(2) bc(2) ac(4) sc(2) 
            parts = []
            idx = 0
            for reg_size in [2, 2, 2, 2, 4, 2]:  # vc, uc, ec, bc, ac, sc
                parts.append(bitstring[idx:idx+reg_size])
                idx += reg_size
        
        if len(parts) == 6:
            v_c, u_c, e_c, b_c, a_c, s_c = parts
            percentage = (count/total_shots)*100
            print(f"s={s_c}, A={a_c}, b={b_c}, e={e_c}, u={u_c}, v={v_c} | Count: {count:4d} ({percentage:5.1f}%)")
            
            # Parse the actual values for verification
            s0, s1 = int(s_c[1]), int(s_c[0])  # s_c = "s1s0"
            a00, a01, a10, a11 = int(a_c[3]), int(a_c[2]), int(a_c[1]), int(a_c[0])  # a_c = "a11a10a01a00"
            b0, b1 = int(b_c[1]), int(b_c[0])  # b_c = "b1b0"
            
            # Verify matrix multiplication: b = A*s
            expected_b0 = (a00 * s0 + a01 * s1) % 2
            expected_b1 = (a10 * s0 + a11 * s1) % 2
            
            print(f"  → s=[{s0},{s1}], A=[[{a00},{a01}],[{a10},{a11}]], b=[{b0},{b1}]")
            print(f"  → Expected b = A*s = [{expected_b0},{expected_b1}] {'✓' if (b0,b1)==(expected_b0,expected_b1) else '✗'}")
    
    print(f"\nCircuit Setup (from code):")
    print(f"- s: Secret key = [1,0] (s0=1, s1=0)")
    print(f"- A: Matrix = [[1,1],[0,1]] (a00=1, a01=1, a10=0, a11=1)")
    print(f"- Expected b = A*s = [[1,1],[0,1]]*[1,0] = [1,0] → measurement: '01'")
    if not add_error:
        print("✓ Without error, b should ALWAYS be [1,0] (measured as '01')!")

# Example usage
if __name__ == "__main__":
    print("Testing WITHOUT error (should be deterministic):")
    print("=" * 50)
    circuit_no_error, results_no_error = simulate_lwe_circuit(add_error=False, shots=1024)
    analyze_lwe_results(results_no_error, add_error=False)
    
    print("\n\nTesting WITH small error:")
    print("=" * 50) 
    circuit_with_error, results_with_error = simulate_lwe_circuit(add_error=True, error_strength=0.1, shots=1024)
    analyze_lwe_results(results_with_error, add_error=True)
    
    # Display the circuit
    print(f"\nQuantum Circuit Structure:")
    print(circuit_no_error.draw())
    circuit_no_error.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'})
    plt.savefig("Simple_circuit_viz/NO-error-circuit.png", dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
    plt.close()

    circuit_with_error.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'})
    plt.savefig("Simple_circuit_viz/WITH-error-circuit.png", dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
    plt.close()