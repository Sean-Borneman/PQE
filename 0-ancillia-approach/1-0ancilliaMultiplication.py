# Fast Quantum Multiplication Algorithm - Qiskit Implementation
# Based on "Fast quantum integer multiplication with zero ancillas"
# by Kahanamoku-Meyer & Yao (arXiv:2403.18006)

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import numpy as np

def qft_manual(circuit, qubits):
    """Manual QFT implementation using basic gates"""
    n = len(qubits)
    for i in range(n):
        circuit.h(qubits[i])
        for j in range(i + 1, n):
            angle = 2 * np.pi / (2 ** (j - i + 1))
            circuit.cp(angle, qubits[j], qubits[i])
    # Swap qubits
    for i in range(n // 2):
        circuit.swap(qubits[i], qubits[n - i - 1])

def qft_inverse_manual(circuit, qubits):
    """Manual inverse QFT implementation using basic gates"""
    n = len(qubits)
    # Swap qubits
    for i in range(n // 2):
        circuit.swap(qubits[i], qubits[n - i - 1])
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            angle = -2 * np.pi / (2 ** (j - i + 1))
            circuit.cp(angle, qubits[j], qubits[i])
        circuit.h(qubits[i])

def phase_product_base_case(circuit, x_qubits, z_qubits, phase_factor):
    """
    Base case PhaseProduct: Apply exp(i * phase_factor * x * z)
    Uses the schoolbook decomposition from Eq. 8 of the paper.
    
    For QFT-based multiplication: applies phase rotations corresponding to
    adding a*x to the output register (in Fourier basis).
    """
    nx = len(x_qubits)
    nz = len(z_qubits)
    
    circuit.barrier(label='PhaseProduct')
    
    # Apply controlled phase rotations for each bit pair
    # Formula: for bit i of input and bit j of output (Fourier mode j),
    # apply phase: 2π * a * 2^i / 2^(j+1) when x[i]=1
    for i in range(nx):
        for j in range(nz):
            # In Draper's formulation for adding in QFT basis
            angle = phase_factor * (2 ** i) / (2 ** (j + 1))
            circuit.cp(angle, x_qubits[i], z_qubits[j])
    
    circuit.barrier()

def quantum_adder_qft(circuit, a_qubits, b_qubits):
    """
    In-place quantum adder: |a⟩|b⟩ → |a⟩|a+b mod 2^n⟩
    Used to compute linear combinations in-place.
    """
    n = min(len(a_qubits), len(b_qubits))
    
    qft_manual(circuit, b_qubits[:n])
    
    for i in range(n):
        for j in range(n - i):
            angle = 2 * np.pi / (2 ** (j + 1))
            circuit.cp(angle, a_qubits[i], b_qubits[i + j])
    
    qft_inverse_manual(circuit, b_qubits[:n])

def phase_product_karatsuba(circuit, x_qubits, z_qubits, phase_factor, depth=0, max_depth=1):
    """
    Recursive PhaseProduct using Karatsuba decomposition (k=2).
    Demonstrates the Toom-Cook polynomial evaluation approach.
    """
    nx = len(x_qubits)
    nz = len(z_qubits)
    
    if depth >= max_depth or nx <= 2 or nz <= 2:
        phase_product_base_case(circuit, x_qubits, z_qubits, phase_factor)
        return
    
    mid_x = nx // 2
    mid_z = nz // 2
    
    x0 = x_qubits[:mid_x]
    x1 = x_qubits[mid_x:]
    z0 = z_qubits[:mid_z]
    z1 = z_qubits[mid_z:]
    
    circuit.barrier(label=f'Karatsuba_L{depth}')
    
    b = 2 ** mid_x
    
    # Product 1: x0 * z0
    phase_product_karatsuba(circuit, x0, z0, phase_factor, depth+1, max_depth)
    
    # Product 2: (x0+x1) * (z0+z1)
    quantum_adder_qft(circuit, x1, x0)
    quantum_adder_qft(circuit, z1, z0)
    
    phase_product_karatsuba(circuit, x0, z0, phase_factor * b, depth+1, max_depth)
    
    quantum_adder_qft(circuit, z1, z0)
    quantum_adder_qft(circuit, x1, x0)
    
    # Product 3: x1 * z1
    phase_product_karatsuba(circuit, x1, z1, phase_factor * (b ** 2), depth+1, max_depth)

def quantum_multiply_classical_value(n_bits, a, use_karatsuba=False):
    """
    Create circuit for U_c×q(a): |x⟩|w⟩ → |x⟩|w + a*x⟩
    
    Algorithm structure:
    1. QFT on output register
    2. PhaseProduct
    3. Inverse QFT
    """
    x_reg = QuantumRegister(n_bits, 'x')
    w_reg = QuantumRegister(2*n_bits, 'w')
    c_reg = ClassicalRegister(2*n_bits, 'c')
    
    circuit = QuantumCircuit(x_reg, w_reg, c_reg)
    
    circuit.barrier(label='QFT')
    qft_manual(circuit, list(w_reg))
    
    phase_factor = 2 * np.pi * a  # Draper-style QFT addition
    
    if use_karatsuba:
        phase_product_karatsuba(circuit, list(x_reg), list(w_reg), phase_factor)
    else:
        phase_product_base_case(circuit, list(x_reg), list(w_reg), phase_factor)
    
    circuit.barrier(label='IQFT')
    qft_inverse_manual(circuit, list(w_reg))
    
    circuit.measure(w_reg, c_reg)
    
    return circuit

def prepare_input_state(circuit, x_reg, value):
    """Prepare input register in computational basis state |value⟩"""
    n = len(x_reg)
    for i in range(n):
        if (value >> i) & 1:
            circuit.x(x_reg[i])

# Example usage and simulation
if __name__ == "__main__":
    print("=" * 70)
    print("Fast Quantum Multiplication Demo")
    print("=" * 70)
    
    n_bits = 3
    a = 4
    x_value = 7 # Input value
    
    print(f"\nCircuit for {n_bits}-bit multiplication by a={a}")
    print(f"Input: |x⟩ = |{x_value}⟩")
    print(f"Expected output: |{x_value}⟩|{a * x_value}⟩")
    print(f"                  (input unchanged, output = {a} × {x_value} = {a * x_value})\n")
    
    # Create circuit
    circuit = quantum_multiply_classical_value(n_bits, a, use_karatsuba=False)
    
    # Prepare input state BEFORE all the gates
    x_reg = circuit.qregs[0]
    prepare_input_state(circuit, x_reg, x_value)
    
    # Move the initialization to the beginning
    # We need to rebuild the circuit with input prep first
    x_reg_new = QuantumRegister(n_bits, 'x')
    w_reg_new = QuantumRegister(2*n_bits, 'w')
    c_reg_new = ClassicalRegister(2*n_bits, 'c')
    
    circuit_full = QuantumCircuit(x_reg_new, w_reg_new, c_reg_new)
    
    # Prepare input
    for i in range(n_bits):
        if (x_value >> i) & 1:
            circuit_full.x(x_reg_new[i])
    
    circuit_full.barrier(label='Input_Prepared')
    
    # Add multiplication circuit
    qft_manual(circuit_full, list(w_reg_new))
    # Phase factor: 2π*a for Draper-style QFT addition
    phase_factor = 2 * np.pi * a
    phase_product_base_case(circuit_full, list(x_reg_new), list(w_reg_new), phase_factor)
    qft_inverse_manual(circuit_full, list(w_reg_new))
    
    circuit_full.measure(w_reg_new, c_reg_new)
    
    print(f"Circuit statistics:")
    print(f"  Total qubits: {circuit_full.num_qubits}")
    print(f"  Input qubits: {n_bits}")
    print(f"  Output qubits: {2*n_bits}")
    print(f"  Ancilla qubits: 0 ✓")
    print(f"  Circuit depth: {circuit_full.depth()}")
    print(f"  Total gates: {sum(circuit_full.count_ops().values())}\n")
    
    print("Gate breakdown:")
    for gate, count in sorted(circuit_full.count_ops().items()):
        print(f"  {gate}: {count}")
    print()
    
    # Run simulation
    print("Running simulation...")
    backend = AerSimulator()
    job = backend.run(circuit_full, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    print("\nMeasurement results (output register only):")
    print("Binary (as measured) -> Reversed -> Decimal : Count")
    print("-" * 60)
    
    # Sort by count
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for bitstring, count in sorted_counts[:10]:  # Show top 10 results
        # Reverse the bitstring to get correct qubit order (Qiskit convention)
        reversed_bitstring = bitstring[::-1]
        decimal_value = int(reversed_bitstring, 2)
        print(f"{bitstring} -> {reversed_bitstring} -> {decimal_value:3d} : {count:4d} times")
    
    # Check if we got the expected result
    expected_output = a * x_value
    exptected_overflow_value = (a * x_value) % (2**n_bits)
    expected_overflow_bitstring = format(exptected_overflow_value, f'0{2*n_bits}b' )
    expected_bitstring = format(expected_output, f'0{2*n_bits}b')
    # Qiskit displays measurements with qubit 0 on the right, so we need to reverse
    expected_measured = expected_bitstring[::-1]
    expected_overflow_bitstring = expected_overflow_bitstring[::-1]
    print(f"\nExpected result: {expected_bitstring} (decimal {expected_output})")
    print(f"As measured:     {expected_measured}")
    
    if expected_measured in counts:
        accuracy = (counts[expected_measured] / 1024) * 100
        print(f"✓ Found with {accuracy:.1f}% probability!")
    
    elif (expected_overflow_bitstring) in counts:
        print(f"If the was overflow (this calc is actually off b/c its mod 2^n not 2^2n so suspect), we'd get {expected_overflow_bitstring}")
        accuracy = (counts[expected_overflow_bitstring] / 1024) * 100
        print(f"✓ Found the mod version {exptected_overflow_value % (2**n_bits)} with {accuracy:.1f}% probability!")
    else:
        print("✗ Expected result not found (try more shots or check for errors)")
    circuit_full.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'})
    plt.savefig("0AncilliaMultiplicationCircuit.png", dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
    plt.close()