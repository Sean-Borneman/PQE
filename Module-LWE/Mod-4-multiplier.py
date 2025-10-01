from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator

class QuantumMod4Multiplier:
    """Scalable quantum circuit builder for multiplication mod 4 using binary multiplication"""
    
    def build_multiplier_circuit(self, val_a, val_b):
        """
        Build quantum circuit for (val_a × val_b) mod 4 using binary multiplication
        
        Algorithm: a × b = a × (b₁×2¹ + b₀×2⁰) = a×b₁×2 + a×b₀
        
        Args:
            val_a (int): First number (0-3)
            val_b (int): Second number (0-3)
            
        Returns:
            QuantumCircuit: Ready-to-run circuit with measurements
        """
        # Create registers
        a = QuantumRegister(2, 'a')           # First number (0-3)
        b = QuantumRegister(2, 'b')           # Second number (0-3) 
        product = QuantumRegister(2, 'prod')  # Final result (0-3)
        temp1 = QuantumRegister(2, 'temp1')   # For a×b₀
        temp2 = QuantumRegister(1, 'temp2')   # For a×b₁×2
        cbits = ClassicalRegister(2, 'result')
        
        circuit = QuantumCircuit(a, b, product, temp1, temp2, cbits)
        
        # Step 1: Initialize inputs
        if val_a & 1: circuit.x(a[0])
        if val_a & 2: circuit.x(a[1])
        if val_b & 1: circuit.x(b[0])
        if val_b & 2: circuit.x(b[1])
        
        circuit.barrier(label='Initialize')
        
        # Step 2: Binary multiplication algorithm
        # Compute a × b₀ (controlled by b[0])
        self._controlled_copy(circuit, b[0], a, temp1)
        
        # Compute a × b₁ × 2 (controlled by b[1], with left shift)
        self._controlled_left_shift_copy(circuit, b[1], a, temp2)

        

        circuit.barrier(label='Partial Products')
        
        # Step 3: Add the partial products: temp1 + temp2 
        self._quantum_add_mod4(circuit, temp1, temp2, product)
        
        circuit.barrier(label='Add Partial Products')
        
        # Step 4: Measure result
        circuit.measure(product, cbits)
        
        return circuit
    
    def _controlled_copy(self, circuit, control, source, target):
        """
        Controlled copy: if control=1, copy source to target
        |control⟩|source⟩|0⟩ → |control⟩|source⟩|control ? source : 0⟩
        """
        circuit.ccx(control, source[0], target[0])
        circuit.ccx(control, source[1], target[1])
    
    def _controlled_left_shift_copy(self, circuit, control, source, target):
        """
        Controlled left shift copy: if control=1, copy (source×2) mod 4 to target
        Left shift by 1 = multiply by 2
        """
        # source×2: bit 0 of source becomes bit 1 of target
        # bit 1 of source wraps to bit 0 of target (since we're mod 4)
        circuit.ccx(control, source[0], target[0])  # source[0] → target[1]
        # circuit.ccx(control, source[1], target[0])  # source[1] → target[0] (wraparound)
    
    def _quantum_add_mod4(self, circuit, a_reg, b_reg, result_reg):
        """
        Add two 2-qubit numbers mod 4: result = (a + b) mod 4
        Reuses the same logic from our adder
        """
        # Copy a to result
        circuit.cx(a_reg[0], result_reg[0])
        circuit.cx(a_reg[1], result_reg[1])
    
        circuit.cx(b_reg[0], result_reg[1])

def run_circuit(circuit, shots=1000):
    """Run a quantum circuit and return results"""
    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    return result.get_counts()

def test_multiplication(val_a, val_b):
    """Test a single multiplication case"""
    multiplier = QuantumMod4Multiplier()
    circuit = multiplier.build_multiplier_circuit(val_a, val_b)
    
    counts = run_circuit(circuit)
    measured_result = int(max(counts, key=counts.get), 2)
    expected = (val_a * val_b) % 4
    
    return measured_result, expected, circuit

def test_all_multiplications():
    """Test all 16 possible multiplications mod 4"""
    multiplier = QuantumMod4Multiplier()
    results = {}
    
    print("Testing all multiplications mod 4 (binary algorithm):")
    print("a × b = result (expected)")
    print("-" * 25)
    
    for a in range(4):
        for b in range(4):
            circuit = multiplier.build_multiplier_circuit(a, b)
            counts = run_circuit(circuit)
            
            measured_result = int(max(counts, key=counts.get), 2)
            expected = (a * b) % 4
            
            status = "✓" if measured_result == expected else "✗"
            print(f"{a} × {b} = {measured_result} ({expected}) {status}")
            
            results[(a,b)] = (measured_result, expected, measured_result == expected)
    
    # Summary
    correct = sum(1 for _, _, success in results.values() if success)
    print(f"\nSuccess rate: {correct}/16 ({100*correct/16:.1f}%)")
    
    return results
    
def visualize_example():
    """Show circuit diagram and trace through algorithm"""
    multiplier = QuantumMod4Multiplier()
    
    print("EXAMPLE: 3 × 2 mod 4 = 2")
    print("=" * 25)
    
    circuit = multiplier.build_multiplier_circuit(3, 2)
    print(circuit.draw())
    print("Algorithm trace:")
    print("  3 × 2 = 3 × (1×2 + 0×1)")
    print("        = 3×1×2 + 3×0×1")
    print("        = 6 + 0")
    print("        = 2 mod 4")
    print()
    
    print(f"Circuit complexity:")
    print(f"  Qubits: {circuit.num_qubits} (vs 6 for lookup table)")
    print(f"  Gates: {len(circuit)}")
    print(f"  Depth: {circuit.depth()}")
    print()
    
    counts = run_circuit(circuit)
    print("Simulation results:")
    for bitstring, count in counts.items():
        decimal_result = int(bitstring, 2)
        print(f"  {bitstring} (={decimal_result}): {count} times")
    
    expected = (3 * 2) % 4
    print(f"\nExpected: {expected}")
    
    return circuit

if __name__ == "__main__":
    print("SCALABLE QUANTUM MOD 4 MULTIPLIER")
    print("=" * 33)
    
    # Show example
    visualize_example()
    
    print()
    
    # Test all cases
    test_all_multiplications()
