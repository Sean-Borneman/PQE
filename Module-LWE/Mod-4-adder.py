from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator

class QuantumMod4Adder:
    """Reusable quantum circuit builder for addition mod 4"""
    
    def build_adder_circuit(self, val_a, val_b):
        """
        Build a complete quantum circuit for (val_a + val_b) mod 4
        
        Args:
            val_a (int): First number (0-3)
            val_b (int): Second number (0-3)
            
        Returns:
            QuantumCircuit: Ready-to-run circuit with measurements
        """
        # Create registers
        a = QuantumRegister(2, 'a')      # First number (0-3)
        b = QuantumRegister(2, 'b')      # Second number (0-3)
        sum_reg = QuantumRegister(2, 'sum')  # Result (0-3)
        cbits = ClassicalRegister(2, 'result')
        
        circuit = QuantumCircuit(a, b, sum_reg, cbits)
        
        # Step 1: Initialize inputs
        if val_a & 1: circuit.x(a[0])  # Set bit 0 if needed
        if val_a & 2: circuit.x(a[1])  # Set bit 1 if needed
        if val_b & 1: circuit.x(b[0])  # Set bit 0 if needed
        if val_b & 2: circuit.x(b[1])  # Set bit 1 if needed
        
        circuit.barrier(label='Initialize')
        
        # Step 2: Quantum addition with carry logic
        circuit.cx(a[0], sum_reg[0])  # Copy a[0] to sum[0]
        circuit.cx(a[1], sum_reg[1])  # Copy a[1] to sum[1]
        circuit.cx(b[0], sum_reg[0])  # Add b[0] to sum[0]
        
        # Carry logic for proper addition
        circuit.x(sum_reg[0])
        circuit.ccx(b[0], sum_reg[0], sum_reg[1])
        circuit.x(sum_reg[0])
        
        circuit.cx(b[1], sum_reg[1])  # Add b[1] to sum[1]
        
        circuit.barrier(label='Add Mod 4')
        
        # Step 3: Measure result
        circuit.measure(sum_reg, cbits)
        
        return circuit

def run_circuit(circuit, shots=1000):
    """Run a quantum circuit and return results"""
    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    return result.get_counts()

def test_addition(val_a, val_b):
    """Test a single addition case"""
    adder = QuantumMod4Adder()
    circuit = adder.build_adder_circuit(val_a, val_b)
    
    counts = run_circuit(circuit)
    measured_result = int(max(counts, key=counts.get), 2)
    expected = (val_a + val_b) % 4
    
    return measured_result, expected, circuit

def test_all_cases():
    """Test all 16 possible additions mod 4"""
    adder = QuantumMod4Adder()
    results = {}
    
    print("Testing all additions mod 4:")
    print("a + b = result (expected)")
    print("-" * 25)
    
    for a in range(4):
        for b in range(4):
            circuit = adder.build_adder_circuit(a, b)
            counts = run_circuit(circuit)
            
            measured_result = int(max(counts, key=counts.get), 2)
            expected = (a + b) % 4
            
            status = "✓" if measured_result == expected else "✗"
            print(f"{a} + {b} = {measured_result} ({expected}) {status}")
            
            results[(a,b)] = (measured_result, expected, measured_result == expected)
    
    # Summary
    correct = sum(1 for _, _, success in results.values() if success)
    print(f"\nSuccess rate: {correct}/16 ({100*correct/16:.1f}%)")
    
    return results

def visualize_example():
    """Show circuit diagram for a specific example"""
    adder = QuantumMod4Adder()
    
    print("EXAMPLE: 2 + 3 mod 4 = 1")
    print("=" * 25)
    
    circuit = adder.build_adder_circuit(2, 3)
    print("Circuit diagram:")
    print(circuit.draw())
    print()
    
    counts = run_circuit(circuit)
    print("Simulation results:")
    for bitstring, count in counts.items():
        decimal_result = int(bitstring, 2)
        print(f"  {bitstring} (={decimal_result}): {count} times")
    
    expected = (2 + 3) % 4
    print(f"\nExpected: {expected}")
    return circuit

def demonstrate_usage():
    """Show how to use the modular adder from other scripts"""
    print("\nUSAGE EXAMPLES:")
    print("=" * 15)
    
    # Example 1: Single calculation
    adder = QuantumMod4Adder()
    circuit = adder.build_adder_circuit(1, 3)
    counts = run_circuit(circuit)
    result = int(max(counts, key=counts.get), 2)
    print(f"1 + 3 mod 4 = {result}")
    
    # Example 2: Use in other calculations
    print(f"Circuit has {circuit.num_qubits} qubits and {len(circuit)} gates")
    
    # Example 3: Get just the circuit without running it
    another_circuit = adder.build_adder_circuit(0, 2)
    print(f"Built circuit for 0 + 2 mod 4 (not executed)")

if __name__ == "__main__":
    print("QUANTUM MOD 4 ADDER")
    print("=" * 20)
    
    # Show example visualization
    visualize_example()
    
    print()
    
    # Test all cases
    test_all_cases()
    
    # Show usage examples
    demonstrate_usage()