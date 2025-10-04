from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np
from QuantumMultiplication import * 

# Example: Compute (3+2x) × (2+3x) in Z_8[x]/(x^2+1)
print("=" * 70)
print("Polynomial Multiplication: (3+2x) × (2+3x) in Z_8[x]/(x^2+1)")
print("=" * 70)

n_bits = 3  # 3 bits for coefficients in Z_8
modulus = 8

# Create registers
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

# Initialize first polynomial: a = 3 + 2x
a0_val = 3  # Constant term
a1_val = 2  # Coefficient of x

for i in range(n_bits):
    if (a0_val >> i) & 1:
        circuit_poly.x(a0_reg[i])
    if (a1_val >> i) & 1:
        circuit_poly.x(a1_reg[i])

# Initialize second polynomial: b = 2 + 3x
b0_val = 2  # Constant term
b1_val = 3  # Coefficient of x

for i in range(n_bits):
    if (b0_val >> i) & 1:
        circuit_poly.x(b0_reg[i])
    if (b1_val >> i) & 1:
        circuit_poly.x(b1_reg[i])

circuit_poly.barrier(label='Inputs_Initialized')

# Now perform polynomial multiplication
# c0 = a0*b0 - a1*b1 (mod 8)
# c1 = a0*b1 + a1*b0 (mod 8)

# Compute c0
QuantumMultiplication.qft_manual(circuit_poly, list(c0_reg))
phase_factor = 2 * np.pi

# Add a0*b0
QuantumMultiplication.phase_triple_product_base_case(circuit_poly, list(a0_reg), list(b0_reg), 
                               list(c0_reg), phase_factor)
# Subtract a1*b1
QuantumMultiplication.phase_triple_product_base_case(circuit_poly, list(a1_reg), list(b1_reg), 
                               list(c0_reg), -phase_factor)

QuantumMultiplication.qft_inverse_manual(circuit_poly, list(c0_reg))

# Compute c1
QuantumMultiplication.qft_manual(circuit_poly, list(c1_reg))

# Add a0*b1
QuantumMultiplication.phase_triple_product_base_case(circuit_poly, list(a0_reg), list(b1_reg), 
                               list(c1_reg), phase_factor)
# Add a1*b0
QuantumMultiplication.phase_triple_product_base_case(circuit_poly, list(a1_reg), list(b0_reg), 
                               list(c1_reg), phase_factor)

QuantumMultiplication.qft_inverse_manual(circuit_poly, list(c1_reg))

# Measure
circuit_poly.measure(c0_reg, m0_reg)
circuit_poly.measure(c1_reg, m1_reg)

print(f"\nInput polynomial 1: {a0_val} + {a1_val}x")
print(f"Input polynomial 2: {b0_val} + {b1_val}x")
print(f"\nManual calculation:")
print(f"  ({a0_val}+{a1_val}x)({b0_val}+{b1_val}x) = {a0_val*b0_val} + {a0_val*b1_val}x + {a1_val*b0_val}x + {a1_val*b1_val}x²")
print(f"                    = {a0_val*b0_val} + {a0_val*b1_val + a1_val*b0_val}x + {a1_val*b1_val}x²")
print(f"                    = {a0_val*b0_val} + {a0_val*b1_val + a1_val*b0_val}x - {a1_val*b1_val}  [x²=-1]")

c0_expected = (a0_val*b0_val - a1_val*b1_val) % modulus
c1_expected = (a0_val*b1_val + a1_val*b0_val) % modulus
print(f"                    = {c0_expected} + {c1_expected}x (mod {modulus})")

print(f"\nCircuit statistics:")
print(f"  Total qubits: {circuit_poly.num_qubits}")
print(f"  Circuit depth: {circuit_poly.depth()}")

# Run simulation
print("\nRunning simulation...")
backend = AerSimulator()
job = backend.run(circuit_poly, shots=1024)
result = job.result()
counts = result.get_counts()

print("\nMeasurement results:")
print("Format: 'c1_bits c0_bits'")
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

for bitstring, count in sorted_counts[:5]:
    # Split into c0 and c1 parts
    c0_bits = bitstring[-2*n_bits:]
    c1_bits = bitstring[:2*n_bits]
    
    # Reverse for correct bit order
    c0_reversed = c0_bits[::-1]
    c1_reversed = c1_bits[::-1]
    
    c0_decimal = int(c0_reversed, 2) % modulus
    c1_decimal = int(c1_reversed, 2) % modulus
    
    print(f"  c0={c0_decimal}, c1={c1_decimal} : {count} times")

print(f"\n✓ Expected: c0={c0_expected}, c1={c1_expected}")
print(f"  Result: {c0_expected} + {c1_expected}x")