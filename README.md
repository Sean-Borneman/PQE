This is a repository of qunatum implementations of post quantum encryption algorithms (PQE)

At the moment I'm mostly workwith LWE related schemes and what you'll see here are a few of my attempt to get quantum advantage:
-One of my first attempts was to use quantum phase estimation which didnt really work becuase theres no useful unitary matrix that we can use to calculate the inner product of A and s nor can we search through a space and make use of a phases ability to store large numbers
-I did breifly try using phase arithmatic but it doesnt save qubits essentially by definition and also doesnt really save gates
- Right now i just have a very basic example running, the only use of quantum phenoena is a phase shift to add error and adjust the rate.scale at which error is added, cool but not really all that useful basicly just bitwise additino/multiplcation on numbers less than 4.

You Cna see a picture of the basic circuit below, basic but it works!
![cirucit_image](https://github.com/Sean-Borneman/PQE/blob/main/Simple_circuit_viz/WITH-error-circuit.png)

There is a horrible script in here that tries to generalize the process for creating theses circuits but it does too much with hybird computing which im trying to avoid. 

My next steps right now are probably to try to scale this or more likely to try to adapt this to Module LWE which is what the NIST finallists use. 

Could also look at NTRU but realistically module lwe should come first and neither will have any kind of quantum advantage with the current approach.
