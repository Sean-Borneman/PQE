#!/usr/bin/env python3
"""
Learning with Errors (LWE) Cryptosystem Implementation using Qiskit
A simplified quantum implementation for educational purposes.

Parameters chosen for simulator compatibility:
- n=2 (secret vector dimension)
- q=8 (modulus, 3 bits)
- Small noise values for demonstration
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from typing import List, Tuple
import random

class QuantumLWE:
    def __init__(self, n: int = 2, q: int = 8, sigma: float = 1.0):
        """
        Initialize LWE parameters
        n: dimension of secret vector
        q: modulus (must be power of 2 for efficient quantum circuits)  
        sigma: noise parameter
        """
        self.n = n
        self.q = q
        self.sigma = sigma
        self.q_bits = int(np.log2(q))  # bits needed to represent mod q
        
    def create_simple_quantum_adder(self, qc: QuantumCircuit, 
                                   a_start: int, 
                                   result_start: int,
                                   temp_start: int) -> None:
        """
        Simple 3-bit quantum adder: result += a (mod 8)
        Direct implementation for q=8 case
        """
        # For 3-bit addition, we implement each bit directly
        # This is more reliable than a general ripple-carry adder
        
        # Save original result bits in temp for carry calculation
        qc.cx(result_start, temp_start)      # temp[0] = result[0]
        qc.cx(result_start + 1, temp_start + 1)  # temp[1] = result[1] 
        qc.cx(result_start + 2, temp_start + 2)  # temp[2] = result[2]
        
        # Bit 0: result[0] = result[0] XOR a[0]
        qc.cx(a_start, result_start)
        
        # Bit 1: result[1] = result[1] XOR a[1] XOR carry0
        # carry0 = temp[0] AND a[0]
        qc.cx(a_start + 1, result_start + 1)  # result[1] ^= a[1]
        qc.ccx(temp_start, a_start, result_start + 1)  # result[1] ^= (temp[0] AND a[0])
        
        # Bit 2: result[2] = result[2] XOR a[2] XOR carry1  
        # carry1 = (temp[1] AND a[1]) XOR (carry0 AND (temp[1] XOR a[1]))
        qc.cx(a_start + 2, result_start + 2)  # result[2] ^= a[2]
        
        # Add carry from bit 1: carry1 = (temp[1] AND a[1])
        qc.ccx(temp_start + 1, a_start + 1, result_start + 2)
        
        # Add propagated carry: carry0 AND (temp[1] XOR a[1])
        # First compute temp[1] XOR a[1] in a spare temp bit
        if temp_start + 3 < qc.num_qubits:
            qc.cx(temp_start + 1, temp_start + 3)  # temp[3] = temp[1] 
            qc.cx(a_start + 1, temp_start + 3)     # temp[3] = temp[1] XOR a[1]
            
            # carry propagation: carry0 AND (temp[1] XOR a[1])
            # We need carry0 in a qubit - use another temp bit
            if temp_start + 4 < qc.num_qubits:
                qc.ccx(temp_start, a_start, temp_start + 4)  # temp[4] = carry0
                qc.ccx(temp_start + 4, temp_start + 3, result_start + 2)  # add propagated carry
                qc.reset(temp_start + 4)
            
            qc.reset(temp_start + 3)
        
        # Clean up temp bits
        qc.reset(temp_start)
        qc.reset(temp_start + 1) 
        qc.reset(temp_start + 2)
        
        # Modular reduction is automatic with 3-bit registers (results >= 8 wrap around)
    
    def create_quantum_adder_mod_q(self, qc: QuantumCircuit, 
                                  a_start: int, 
                                  result_start: int,
                                  temp_start: int) -> None:
        """
        Wrapper for the simple adder
        """
        self.create_simple_quantum_adder(qc, a_start, result_start, temp_start)
    
    def create_inner_product_circuit(self, a_vec: List[int], s_vec: List[int]) -> QuantumCircuit:
        """
        Create quantum circuit to compute inner product <a,s> mod q
        """
        # Simplified qubit allocation for the cleaner adder:
        qubits_per_element = self.q_bits
        temp_qubits = 8  # More temp space for the cleaner arithmetic
        total_qubits = (self.n * 2 * qubits_per_element +  # a and s vectors
                       temp_qubits +                       # temp space for arithmetic
                       self.q_bits)                        # result
        
        qc = QuantumCircuit(total_qubits, self.q_bits)
        
        # Define qubit layout
        # a[0], a[1], ..., a[n-1], s[0], s[1], ..., s[n-1], temp[0-7], result[0-2]
        a_start = 0
        s_start = self.n * qubits_per_element
        temp_start = self.n * 2 * qubits_per_element
        result_start = temp_start + temp_qubits
        
        # Initialize input values
        for i in range(self.n):
            for bit in range(self.q_bits):
                # Initialize a[i]
                if (a_vec[i] >> bit) & 1:
                    qc.x(a_start + i * qubits_per_element + bit)
                # Initialize s[i]  
                if (s_vec[i] >> bit) & 1:
                    qc.x(s_start + i * qubits_per_element + bit)
        
        # Compute inner product: sum over i of a[i] * s[i]
        # For simplicity, we'll use repeated addition for small values
        for i in range(self.n):
            s_val = s_vec[i]
            if s_val > 0:
                # Add a[i] to result s_val times (multiplication by repeated addition)
                for _ in range(min(s_val, 4)):  # Limit iterations to keep circuit manageable
                    # Copy a[i] to temp space
                    for bit in range(self.q_bits):
                        qc.cx(a_start + i * qubits_per_element + bit, temp_start + bit)
                    
                    # Add temp to result: result += temp
                    self.create_quantum_adder_mod_q(
                        qc,
                        temp_start,                         # source (a[i] copy)
                        result_start,                       # destination (result)  
                        temp_start + 3                      # temp space for arithmetic
                    )
                    
                    # Clear temp space for next iteration  
                    for bit in range(self.q_bits):
                        qc.cx(a_start + i * qubits_per_element + bit, temp_start + bit)
        
        # Measure final result
        for i in range(self.q_bits):
            qc.measure(result_start + i, i)
            
        return qc
    
    def generate_lwe_samples(self, m: int, secret: List[int]) -> List[Tuple[List[int], int]]:
        """
        Generate m LWE samples (a_i, b_i) where b_i = <a_i, s> + e_i mod q
        """
        samples = []
        
        for _ in range(m):
            # Generate random vector a
            a = [random.randint(0, self.q-1) for _ in range(self.n)]
            
            # Compute inner product <a, s>
            inner_prod = sum(a[i] * secret[i] for i in range(self.n)) % self.q
            
            # Add small noise
            noise = random.randint(0, 2)  # Small noise for demo
            b = (inner_prod + noise) % self.q
            
            samples.append((a, b))
            
        return samples
    
    def quantum_lwe_keygen(self) -> Tuple[List[int], List[Tuple[List[int], int]]]:
        """
        Generate LWE key pair
        Returns: (secret_key, public_key_samples)
        """
        # Generate random secret
        secret = [random.randint(1, self.q//2) for _ in range(self.n)]
        
        # Generate public key samples
        m = 4  # Number of samples (small for demo)
        public_samples = self.generate_lwe_samples(m, secret)
        
        print(f"Secret key: {secret}")
        print(f"Public key samples: {public_samples}")
        
        return secret, public_samples
    
    def quantum_lwe_encrypt(self, message: int, public_samples: List[Tuple[List[int], int]]) -> Tuple[List[int], int]:
        """
        Encrypt a message using LWE
        """
        # Select random subset of public samples
        selected = random.sample(public_samples, len(public_samples)//2 + 1)
        
        # Compute u = sum of selected a vectors
        u = [0] * self.n
        for (a, _) in selected:
            for i in range(self.n):
                u[i] = (u[i] + a[i]) % self.q
        
        # Compute v = sum of selected b values + encoded message
        v = sum(b for (_, b) in selected) % self.q
        encoded_message = message * (self.q // 4)  # Simple encoding
        v = (v + encoded_message) % self.q
        
        return u, v
    
    def quantum_lwe_decrypt(self, ciphertext: Tuple[List[int], int], secret: List[int]) -> int:
        """
        Decrypt using quantum inner product computation
        """
        u, v = ciphertext
        
        # Create and run quantum circuit to compute <u, s>
        qc = self.create_inner_product_circuit(u, secret)
        
        # Simulate the circuit
        simulator = AerSimulator()
        compiled_circuit = transpile(qc, simulator)
        result = simulator.run(compiled_circuit, shots=1024).result()
        counts = result.get_counts()
        
        # Get most frequent result
        inner_product = int(max(counts.keys(), key=counts.get), 2)
        
        # Decrypt: v - <u,s> mod q  
        decrypted_encoded = (v - inner_product) % self.q
        
        # Decode message (simple threshold decoding)
        decoded_message = 1 if decrypted_encoded > self.q // 2 else 0
        
        print(f"Quantum inner product result: {inner_product}")
        print(f"Decrypted encoded value: {decrypted_encoded}")
        
        return decoded_message

def main():
    """
    Demonstrate the quantum LWE cryptosystem
    """
    print("=== Quantum LWE Cryptosystem Demo ===\n")
    
    # Initialize LWE system
    lwe = QuantumLWE(n=2, q=8, sigma=1.0)
    
    # Key generation
    print("1. Key Generation:")
    secret_key, public_key = lwe.quantum_lwe_keygen()
    
    # Encryption
    print("\n2. Encryption:")
    message = 1
    print(f"Original message: {message}")
    ciphertext = lwe.quantum_lwe_encrypt(message, public_key)
    print(f"Ciphertext: {ciphertext}")
    
    # Decryption
    print("\n3. Quantum Decryption:")
    decrypted = lwe.quantum_lwe_decrypt(ciphertext, secret_key)
    print(f"Decrypted message: {decrypted}")
    print(f"Decryption {'SUCCESS' if decrypted == message else 'FAILED'}!")
    
    # Test inner product circuit separately
    print("\n4. Testing Inner Product Circuit:")
    test_a = [3, 5]
    test_s = [2, 1] 
    expected = (3*2 + 5*1) % 8  # = 11 % 8 = 3
    print(f"Testing <{test_a}, {test_s}> mod 8")
    print(f"Expected result: {expected}")
    
    qc = lwe.create_inner_product_circuit(test_a, test_s)
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=1024).result()
    counts = result.get_counts()
    
    print(f"Quantum circuit results: {counts}")
    most_frequent = max(counts.keys(), key=counts.get)
    quantum_result = int(most_frequent, 2)
    print(f"Most frequent result: {quantum_result}")
    print(f"Inner product test {'PASSED' if quantum_result == expected else 'FAILED'}!")
    
    # VISUALIZATION OPTIONS
    print("\n5. Circuit Visualization:")
    
    # Option 1: Text-based circuit diagram
    print("Text representation:")
    print(qc.draw())
    
    # Option 2: Save circuit diagram as image (requires matplotlib)

    from qiskit.visualization import circuit_drawer
    circuit_img = circuit_drawer(qc, output='mpl', style='iqx')
    circuit_img.savefig('lwe_quantum_circuit.png', dpi=300, bbox_inches='tight')
    print("Circuit diagram saved as 'lwe_quantum_circuit.png'")

    # Option 3: Interactive circuit viewer (if in Jupyter)
    try:
        from qiskit.visualization import plot_circuit_layout
        print("For interactive viewing, run in Jupyter notebook:")
        print("qc.draw('mpl')  # Shows interactive plot")
    except:
        pass
    
    # Option 4: Circuit statistics
    print(f"\nCircuit Statistics:")
    print(f"Total qubits: {qc.num_qubits}")
    print(f"Circuit depth: {qc.depth()}")
    print(f"Gate count: {qc.count_ops()}")
    
    # Option 5: Show gate decomposition
    print("\nGate decomposition:")
    decomposed = qc.decompose()
    print(f"After decomposition: {decomposed.count_ops()}")

if __name__ == "__main__":
    main()