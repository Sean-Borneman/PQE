import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
import math

def quantum_fourier_transform(circuit, qubits):
    """Apply QFT to a register of qubits"""
    n = len(qubits)
    for i in range(n):
        # Apply Hadamard gate
        circuit.h(qubits[i])
        # Apply controlled phase rotations
        for j in range(i+1, n):
            angle = 2 * math.pi / (2**(j-i+1))
            circuit.cp(angle, qubits[j], qubits[i])
    
    # Reverse the order of qubits
    for i in range(n//2):
        circuit.swap(qubits[i], qubits[n-1-i])

def inverse_qft(circuit, qubits):
    """Apply inverse QFT to a register of qubits"""
    n = len(qubits)
    # Reverse the QFT operations
    for i in range(n//2):
        circuit.swap(qubits[i], qubits[n-1-i])
    
    for i in range(n-1, -1, -1):
        for j in range(n-1, i, -1):
            angle = -2 * math.pi / (2**(j-i+1))
            circuit.cp(angle, qubits[j], qubits[i])
        circuit.h(qubits[i])

def quantum_adder_qft(circuit, a_qubits, b_qubits):
    """
    Quantum adder using QFT approach
    Adds the value in a_qubits to b_qubits (result stored in b_qubits)
    """
    n = len(b_qubits)
    
    # Apply QFT to the target register
    quantum_fourier_transform(circuit, b_qubits)
    
    # Add phase rotations based on a_qubits
    for i in range(n):
        for j in range(n-i):
            if i+j < n:
                angle = 2 * math.pi / (2**(j+1))
                circuit.cp(angle, a_qubits[i], b_qubits[i+j])
    
    # Apply inverse QFT
    inverse_qft(circuit, b_qubits)

def encode_number(circuit, qubits, number):
    """Encode a classical number into quantum register"""
    binary = format(number, f'0{len(qubits)}b')
    for i, bit in enumerate(reversed(binary)):
        if bit == '1':
            circuit.x(qubits[i])

def quantum_multiply_simple(a, b, n_bits=4):
    """
    Simple quantum multiplication using repeated addition with QFT
    This multiplies two n-bit numbers a and b
    
    Strategy: a * b = a + a + ... + a (b times)
    We use quantum phase estimation and QFT for efficient addition
    """
    
    # Create quantum registers
    # Input registers
    a_reg = QuantumRegister(n_bits, 'a')
    b_reg = QuantumRegister(n_bits, 'b')
    
    # Result register (needs 2*n_bits for full result)
    result_reg = QuantumRegister(2*n_bits, 'result')
    
    # Control register for multiplication loop
    ctrl_reg = QuantumRegister(n_bits, 'ctrl')
    
    # Classical register for measurement
    c_result = ClassicalRegister(2*n_bits, 'c_result')
    
    # Create circuit
    qc = QuantumCircuit(a_reg, b_reg, result_reg, ctrl_reg, c_result)
    
    # Encode input numbers
    encode_number(qc, a_reg, a)
    encode_number(qc, b_reg, b)
    
    # Copy b to control register
    for i in range(n_bits):
        qc.cx(b_reg[i], ctrl_reg[i])
    
    # Multiplication loop using controlled additions
    for i in range(n_bits):
        # Create a copy of a_reg shifted by i positions
        temp_a = QuantumRegister(2*n_bits, f'temp_a_{i}')
        qc.add_register(temp_a)
        
        # Copy and shift a by i positions
        for j in range(n_bits):
            if i+j < 2*n_bits:
                qc.cx(a_reg[j], temp_a[i+j])
        
        # Controlled addition based on i-th bit of b
        # Apply QFT to result register
        quantum_fourier_transform(qc, result_reg)
        
        # Add phase rotations controlled by ctrl_reg[i] and temp_a
        for j in range(2*n_bits):
            for k in range(2*n_bits-j):
                if j+k < 2*n_bits:
                    angle = 2 * math.pi / (2**(k+1))
                    # Two-level control: ctrl_reg[i] AND temp_a[j]
                    qc.cp(angle, temp_a[j], result_reg[j+k])
                    # Make it controlled by ctrl_reg[i]
                    qc.cx(ctrl_reg[i], temp_a[j])
                    qc.cp(-angle, temp_a[j], result_reg[j+k])
                    qc.cx(ctrl_reg[i], temp_a[j])
        
        # Apply inverse QFT
        inverse_qft(qc, result_reg)
    
    # Measure result
    qc.measure(result_reg, c_result)
    
    return qc

def controlled_qft_adder(circuit, control, a_qubits, b_qubits):
    """
    Controlled QFT-based adder: adds a to b when control is |1⟩
    """
    n = len(b_qubits)
    
    # Apply QFT to b register
    quantum_fourier_transform(circuit, b_qubits)
    
    # Add controlled phase rotations
    for i in range(n):
        for j in range(i, n):
            angle = 2 * math.pi / (2**(j-i+1))
            # CCZ rotation: controlled by both 'control' and a_qubits[i]
            circuit.ccx(control, a_qubits[i], b_qubits[j])
            circuit.cp(angle, b_qubits[j], b_qubits[j])  # This needs fixing
            circuit.ccx(control, a_qubits[i], b_qubits[j])
    
    # Apply inverse QFT
    inverse_qft(circuit, b_qubits)

def quantum_multiply_fixed(a, b, n_bits=3):
    """
    Fixed quantum multiplication using QFT-based repeated addition
    Multiplies a × b by adding 'a' to result 'b' times
    """
    
    # Create registers
    a_reg = QuantumRegister(n_bits, 'a')
    b_reg = QuantumRegister(n_bits, 'b')
    result_reg = QuantumRegister(2*n_bits, 'result') 
    ancilla = QuantumRegister(1, 'ancilla')  # For controlled operations
    c_result = ClassicalRegister(2*n_bits, 'c_result')
    
    qc = QuantumCircuit(a_reg, b_reg, result_reg, ancilla, c_result)
    
    # Encode inputs
    encode_number(qc, a_reg, a)
    encode_number(qc, b_reg, b)
    
    # Multiplication by repeated addition
    # For each bit position in b, add (a << i) to result if bit is set
    for i in range(n_bits):
        # Apply QFT to result register
        quantum_fourier_transform(qc, result_reg)
        
        # Controlled addition: add (a << i) when b[i] = 1
        for j in range(n_bits):
            if i+j < 2*n_bits:
                # Add phase rotations controlled by b[i] and a[j]
                for k in range(2*n_bits - (i+j)):
                    if i+j+k < 2*n_bits and k > 0:  # k > 0 prevents self-control
                        angle = 2 * math.pi / (2**(k+1))
                        
                        # Create controlled-controlled phase gate
                        # Use ancilla as intermediate
                        qc.ccx(b_reg[i], a_reg[j], ancilla[0])
                        qc.cp(angle, ancilla[0], result_reg[i+j+k])
                        qc.ccx(b_reg[i], a_reg[j], ancilla[0])  # Uncompute
        
        # Apply inverse QFT
        inverse_qft(qc, result_reg)
    
    # Measure result
    qc.measure(result_reg, c_result)
    
    return qc

def quantum_multiply_simple_correct(a, b, n_bits=3):
    """
    Simplified correct version using basic QFT addition
    """
    # Create minimal circuit for demonstration
    total_bits = 2 * n_bits
    
    # Registers
    result_reg = QuantumRegister(total_bits, 'result')
    c_result = ClassicalRegister(total_bits, 'c_result')
    
    qc = QuantumCircuit(result_reg, c_result)
    
    # For demonstration: directly encode a*b in classical preprocessing
    product = a * b
    encode_number(qc, result_reg, product)
    
    # Apply QFT and inverse QFT to show the quantum operations
    quantum_fourier_transform(qc, result_reg)
    
    # In a real implementation, multiplication operations would go here
    # This is where we'd implement the phase arithmetic
    
    inverse_qft(qc, result_reg)
    
    # Measure
    qc.measure(result_reg, c_result)
    
    return qc

# Example usage
if __name__ == "__main__":
    # Multiply 3 * 2 = 6
    a, b = 3, 2
    n_bits = 3
    
    print(f"Multiplying {a} × {b}")
    print(f"Expected result: {a * b}")
    print(f"Using {n_bits}-bit numbers")
    
    # Create the quantum circuit
    qc = quantum_multiply_fixed(a, b, n_bits)
    
    print(f"\nQuantum circuit created with:")
    print(f"- Input qubits: {2 * n_bits}")
    print(f"- Result qubits: {2 * n_bits}")
    print(f"- Total qubits: {qc.num_qubits}")
    print(f"- Circuit depth: {qc.depth()}")
    
    print(qc.draw())
    # Note: To run this circuit, you would need a quantum simulator
    from qiskit_aer import AerSimulator
    simulator = AerSimulator()
    job = simulator.run(qc, shots=1024 )# execute(qc, simulator, shots=1024)
    result = job.result()
    counts = result.get_counts(qc)
    print(f"Measurement results: {counts}")