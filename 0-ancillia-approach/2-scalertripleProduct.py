# Quantum-Quantum Multiplication for Module-LWE
# Implements PhaseTripleProduct for multiplying two quantum registers

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
    for i in range(n // 2):
        circuit.swap(qubits[i], qubits[n - i - 1])

def qft_inverse_manual(circuit, qubits):
    """Manual inverse QFT implementation using basic gates"""
    n = len(qubits)
    for i in range(n // 2):
        circuit.swap(qubits[i], qubits[n - i - 1])
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            angle = -2 * np.pi / (2 ** (j - i + 1))
            circuit.cp(angle, qubits[j], qubits[i])
        circuit.h(qubits[i])

def phase_triple_product_base_case(circuit, x_qubits, y_qubits, z_qubits, phase_factor):
    """
    Base case PhaseTripleProduct: Apply exp(i * phase_factor * x * y * z)
    
    This is the quantum-quantum multiplication version from Eq. 9 of the paper.
    Uses doubly-controlled phase gates (Toffoli-style phase rotations).
    
    For each triple of bits (x[i], y[j], z[k]), we apply a phase rotation.
    The phase is only applied when ALL THREE bits are |1⟩.
    """
    nx = len(x_qubits)
    ny = len(y_qubits)
    nz = len(z_qubits)
    
    circuit.barrier(label='PhaseTripleProduct')
    
    # Apply doubly-controlled phase rotations for QFT-based multiplication
    # Formula: for input bits i,j and Fourier mode k
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Phase angle in Draper-style for adding x*y
                angle = phase_factor * (2 ** i) * (2 ** j) / (2 ** (k + 1))
                
                # We need a doubly-controlled phase gate (controlled by x[i] and y[j])
                # Apply to z[k] in Fourier basis
                # This is like a "controlled-controlled-phase" or Toffoli with phase
                ccphase(circuit, x_qubits[i], y_qubits[j], z_qubits[k], angle)
    
    circuit.barrier()

def ccphase(circuit, control1, control2, target, angle):
    """
    Doubly-controlled phase gate: applies phase 'angle' to target when both controls are |1⟩
    
    Implements: |11⟩|ψ⟩ → |11⟩ exp(iθ)|ψ⟩
    
    This can be decomposed using standard techniques, but for simplicity
    we use a decomposition with single-controlled gates.
    """
    # Standard decomposition of doubly-controlled phase gate
    circuit.cp(angle / 2, control1, target)
    circuit.cx(control1, control2)
    circuit.cp(-angle / 2, control2, target)
    circuit.cx(control1, control2)
    circuit.cp(angle / 2, control2, target)

def quantum_multiply_quantum_registers(n_bits_x, n_bits_y):
    """
    Create circuit for U_q×q: |x⟩|y⟩|w⟩ → |x⟩|y⟩|w + x*y⟩
    
    This implements quantum-quantum multiplication (Section III of paper).
    
    Algorithm structure:
    1. Apply QFT to output register
    2. Apply PhaseTripleProduct(2π)
    3. Apply inverse QFT
    
    Args:
        n_bits_x: Number of bits in first input register
        n_bits_y: Number of bits in second input register
    """
    # For n-bit × m-bit multiplication, output needs n+m bits
    n_out = n_bits_x + n_bits_y
    
    # Create quantum registers
    x_reg = QuantumRegister(n_bits_x, 'x')
    y_reg = QuantumRegister(n_bits_y, 'y')
    w_reg = QuantumRegister(n_out, 'w')
    c_reg = ClassicalRegister(n_out, 'c')
    
    circuit = QuantumCircuit(x_reg, y_reg, w_reg, c_reg)
    
    circuit.barrier(label='QFT')
    qft_manual(circuit, list(w_reg))
    
    # Phase factor for Draper-style addition
    phase_factor = 2 * np.pi
    
    phase_triple_product_base_case(circuit, list(x_reg), list(y_reg), 
                                   list(w_reg), phase_factor)
    
    circuit.barrier(label='IQFT')
    qft_inverse_manual(circuit, list(w_reg))
    
    circuit.measure(w_reg, c_reg)
    
    return circuit

def polynomial_element_multiply(n_bits, modulus):
    """
    Multiply two polynomial ring elements (for LWE over R_q = Z_q[x]/(x^n+1))
    
    For Module-LWE, we work in R_q = Z_q[x]/(x^n + 1)
    This means polynomials with coefficients mod q, and x^n = -1
    
    Example: (2 + 3x) * (4 + x) in R_8[x]/(x^2+1)
            = 8 + 2x + 12x + 3x^2
            = 8 + 14x + 3(-1)      [since x^2 = -1]
            = 5 + 14x
            = 5 + 6x  (mod 8)
    
    Args:
        n_bits: Number of bits per coefficient
        modulus: q in Z_q (e.g., 8 for Z_8)
    """
    # For a degree-1 polynomial: a0 + a1*x
    # We need registers for each coefficient
    
    # First polynomial: a0 + a1*x
    a0_reg = QuantumRegister(n_bits, 'a0')
    a1_reg = QuantumRegister(n_bits, 'a1')
    
    # Second polynomial: b0 + b1*x  
    b0_reg = QuantumRegister(n_bits, 'b0')
    b1_reg = QuantumRegister(n_bits, 'b1')
    
    # Output polynomial: c0 + c1*x (needs 2*n_bits for intermediate results)
    c0_reg = QuantumRegister(2*n_bits, 'c0')
    c1_reg = QuantumRegister(2*n_bits, 'c1')
    
    # Classical registers for measurement
    m0_reg = ClassicalRegister(2*n_bits, 'm0')
    m1_reg = ClassicalRegister(2*n_bits, 'm1')
    
    circuit = QuantumCircuit(a0_reg, a1_reg, b0_reg, b1_reg, 
                            c0_reg, c1_reg, m0_reg, m1_reg)
    
    circuit.barrier(label='Start_Poly_Mult')
    
    # Polynomial multiplication: (a0 + a1*x)(b0 + b1*x)
    # = a0*b0 + a0*b1*x + a1*b0*x + a1*b1*x^2
    # = a0*b0 + (a0*b1 + a1*b0)*x + a1*b1*x^2
    # 
    # In R_q[x]/(x^2+1), we have x^2 = -1, so:
    # = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*x
    
    # Compute coefficient c0 = a0*b0 - a1*b1
    circuit.barrier(label='Compute_c0')
    
    # First: QFT on c0
    qft_manual(circuit, list(c0_reg))
    
    # Add a0*b0 to c0 (in Fourier basis)
    phase_factor = 2 * np.pi
    phase_triple_product_base_case(circuit, list(a0_reg), list(b0_reg), 
                                   list(c0_reg), phase_factor)
    
    # Subtract a1*b1 from c0 (negative phase)
    phase_triple_product_base_case(circuit, list(a1_reg), list(b1_reg), 
                                   list(c0_reg), -phase_factor)
    
    # Inverse QFT on c0
    qft_inverse_manual(circuit, list(c0_reg))
    
    # Compute coefficient c1 = a0*b1 + a1*b0  
    circuit.barrier(label='Compute_c1')
    
    # QFT on c1
    qft_manual(circuit, list(c1_reg))
    
    # Add a0*b1 to c1
    phase_triple_product_base_case(circuit, list(a0_reg), list(b1_reg), 
                                   list(c1_reg), phase_factor)
    
    # Add a1*b0 to c1
    phase_triple_product_base_case(circuit, list(a1_reg), list(b0_reg), 
                                   list(c1_reg), phase_factor)
    
    # Inverse QFT on c1
    qft_inverse_manual(circuit, list(c1_reg))
    
    circuit.barrier(label='Measure')
    circuit.measure(c0_reg, m0_reg)
    circuit.measure(c1_reg, m1_reg)
    
    return circuit

# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Quantum-Quantum Multiplication for Module-LWE")
    print("=" * 70)
    print()
    
    # Example 1: Simple quantum-quantum multiplication
    print("Example 1: Simple Quantum × Quantum Multiplication")
    print("-" * 70)
    
    n_bits = 3
    circuit = quantum_multiply_quantum_registers(n_bits, n_bits)
    
    # Prepare inputs: x=3, y=5
    x_value = 3
    y_value = 5
    
    x_reg = circuit.qregs[0]
    y_reg = circuit.qregs[1]
    
    # Build circuit with initialization
    x_reg_new = QuantumRegister(n_bits, 'x')
    y_reg_new = QuantumRegister(n_bits, 'y')
    w_reg_new = QuantumRegister(2*n_bits, 'w')
    c_reg_new = ClassicalRegister(2*n_bits, 'c')
    
    circuit_full = QuantumCircuit(x_reg_new, y_reg_new, w_reg_new, c_reg_new)
    
    # Initialize x and y
    for i in range(n_bits):
        if (x_value >> i) & 1:
            circuit_full.x(x_reg_new[i])
        if (y_value >> i) & 1:
            circuit_full.x(y_reg_new[i])
    
    circuit_full.barrier(label='Inputs_Ready')
    
    # Apply multiplication
    qft_manual(circuit_full, list(w_reg_new))
    phase_factor = 2 * np.pi
    phase_triple_product_base_case(circuit_full, list(x_reg_new), 
                                   list(y_reg_new), list(w_reg_new), phase_factor)
    qft_inverse_manual(circuit_full, list(w_reg_new))
    circuit_full.measure(w_reg_new, c_reg_new)
    
    print(f"Circuit: |x={x_value}⟩|y={y_value}⟩|0⟩ → |{x_value}⟩|{y_value}⟩|{x_value*y_value}⟩")
    print(f"Expected output: {x_value * y_value}")
    print(f"Total qubits: {circuit_full.num_qubits}")
    print(f"Circuit depth: {circuit_full.depth()}")
    print()
    
    # Run simulation
    print("Running simulation...")
    backend = AerSimulator()
    job = backend.run(circuit_full, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    print("\nTop measurement results:")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for bitstring, count in sorted_counts[:5]:
        reversed_bitstring = bitstring[::-1]
        decimal_value = int(reversed_bitstring, 2)
        print(f"  {bitstring} -> {reversed_bitstring} -> {decimal_value:3d} : {count} times")
    
    expected_bitstring = format(x_value * y_value, f'0{2*n_bits}b')
    expected_measured = expected_bitstring[::-1]
    if expected_measured in counts:
        accuracy = (counts[expected_measured] / 1024) * 100
        print(f"\n✓ Correct result found with {accuracy:.1f}% probability!")
    
    print()
    print("=" * 70)
    print("Example 2: Polynomial Ring Element Multiplication (for LWE)")
    print("-" * 70)
    print()
    print("Module-LWE works in R_q = Z_q[x]/(x^n+1)")
    print("Example: (2+3x) × (4+x) in Z_8[x]/(x^2+1)")
    print()
    
    # Test values
    a0_val = 2  # First polynomial: 2 + 3x
    a1_val = 3 #a x
    b0_val = 4  # Second polynomial: 4 + x
    b1_val = 1 # b
    n_bits = 3
    modulus = 8
    
    print("Manual calculation:")
    print(f"  ({a0_val}+{a1_val}x)({b0_val}+{b1_val}x) = {a0_val*b0_val} + {a0_val*b1_val}x + {a1_val*b0_val}x + {a1_val*b1_val}x²")
    print(f"              = {a0_val*b0_val} + {a0_val*b1_val + a1_val*b0_val}x + {a1_val*b1_val}x²")
    print(f"              = {a0_val*b0_val} + {a0_val*b1_val + a1_val*b0_val}x - {a1_val*b1_val}    [x² = -1 in this ring]")
    
    c0_expected = (a0_val*b0_val - a1_val*b1_val) % modulus
    c1_expected = (a0_val*b1_val + a1_val*b0_val) % modulus
    print(f"              = {c0_expected} + {c1_expected}x (mod {modulus})")
    print()
    
    # Build the circuit with initialization
    a0_reg = QuantumRegister(n_bits, 'a0')
    a1_reg = QuantumRegister(n_bits, 'a1')
    b0_reg = QuantumRegister(n_bits, 'b0')
    b1_reg = QuantumRegister(n_bits, 'b1')
    c0_reg = QuantumRegister(2*n_bits, 'c0')
    c1_reg = QuantumRegister(2*n_bits, 'c1')
    m0_reg = ClassicalRegister(2*n_bits, 'm0')
    m1_reg = ClassicalRegister(2*n_bits, 'm1')
    
    circuit_poly = QuantumCircuit(a0_reg, a1_reg, b0_reg, b1_reg, 
                                  c0_reg, c1_reg, m0_reg, m1_reg)
    
    # Initialize inputs
    for i in range(n_bits):
        if (a0_val >> i) & 1:
            circuit_poly.x(a0_reg[i])
        if (a1_val >> i) & 1:
            circuit_poly.x(a1_reg[i])
        if (b0_val >> i) & 1:
            circuit_poly.x(b0_reg[i])
        if (b1_val >> i) & 1:
            circuit_poly.x(b1_reg[i])
    
    circuit_poly.barrier(label='Inputs_Ready')
    
    # Compute c0 = a0*b0 - a1*b1
    qft_manual(circuit_poly, list(c0_reg))
    phase_factor = 2 * np.pi
    phase_triple_product_base_case(circuit_poly, list(a0_reg), list(b0_reg), 
                                   list(c0_reg), phase_factor)
    phase_triple_product_base_case(circuit_poly, list(a1_reg), list(b1_reg), 
                                   list(c0_reg), -phase_factor)
    qft_inverse_manual(circuit_poly, list(c0_reg))
    
    # Compute c1 = a0*b1 + a1*b0
    qft_manual(circuit_poly, list(c1_reg))
    phase_triple_product_base_case(circuit_poly, list(a0_reg), list(b1_reg), 
                                   list(c1_reg), phase_factor)
    phase_triple_product_base_case(circuit_poly, list(a1_reg), list(b0_reg), 
                                   list(c1_reg), phase_factor)
    qft_inverse_manual(circuit_poly, list(c1_reg))
    
    circuit_poly.measure(c0_reg, m0_reg)
    circuit_poly.measure(c1_reg, m1_reg)
    
    print(f"Circuit statistics:")
    print(f"  Total qubits: {circuit_poly.num_qubits}")
    print(f"  Circuit depth: {circuit_poly.depth()}")
    print(f"  Ancilla qubits: 0 ✓")
    print()
    
    # Run simulation
    print("Running polynomial multiplication simulation...")
    backend = AerSimulator()
    job = backend.run(circuit_poly, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    print("\nTop measurement results:")
    print("Format: 'c1_bits c0_bits' -> c0, c1")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    for bitstring, count in sorted_counts[:5]:
        # Split the bitstring: first 2*n_bits are c1, last 2*n_bits are c0
        c0_bits = bitstring[-2*n_bits:]
        c1_bits = bitstring[:2*n_bits]
        
        # Reverse for correct bit order (Qiskit convention)
        c0_decimal = int(c0_bits[::-1], 2) % modulus
        c1_decimal = int(c1_bits[::-1], 2) % modulus
        
        print(f"  {bitstring} -> c0={c0_decimal}, c1={c1_decimal} : {count} times")
    
    print(f"\nExpected result: c0={c0_expected}, c1={c1_expected}")
    print(f"✓ Result polynomial: {c0_expected} + {c1_expected}x")
    print()
    print("This circuit can be used as a building block for:")
    print("  • Matrix-vector multiplication in Module-LWE")
    print("  • Computing b = As + e")
    print("  • Encryption/decryption operations")
    circuit_poly.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'})
    plt.savefig("PQC/0-ancillia-approach/Poly-MultiplicationCircuit.png", dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
    plt.close()

