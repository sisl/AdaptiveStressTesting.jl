# AdaptiveStressTesting.jl

Maintainer: Ritchie Lee, Carnegie Mellon University Silicon Valley, ritchie.lee@sv.cmu.edu

Adaptive Stress Testing is a stress testing tool for finding failure occurrences in multiple time step simulations.  The algorithm uses Monte Carlo tree search, a state-of-the-art planning algorithm, to adapt sampling during search.  This approach leads to efficient best-first exploration of the search space.  The black box method does not make any assumptions about the internal details of the system under test.  Adaptive Stress Testing has been previously applied to stress test airborne collision avoidance systems searching for near mid-air collisions in simulated aircraft encounters [1].   

## Usage

The recommended approach is for the user to define a custom type to contain the simulation state and parameters.  The adaptive stress testing tool then requires the user to expose three functions to the user simulator.
* ``initialize(sim)`` - Resets the simulator to the initial state
* ``update(sim)`` - Steps the simulator forward a single time step.  The tuple ``(prob, isevent, dist)`` is returned, where prob is the probability of taking that step, isevent indicates whether the failure event has occurred, and dist is an optional distance metric that hints to the optimizer how close the execution was to an event.
* ``isterminal(sim)`` - Returns true if the simulation has ended, false otherwise. 

These functions, along with configuration parameters, should be passed to create the adaptive stress test object 
```julia
ast = AdaptiveStressTest(ast_params, sim, MySimType.initialize, MySimType.update, MySimType.isterminal)
```

To draw Monte Carlo samples from the simulator:
```julia
sample(ast)
```
To draw N samples:
```julia
sample(ast, N)
```
Monte Carlo sampling is useful for testing and debugging your simulator before you run the actual stress test. 

When you're ready, run the stess test:
```julia
result = stress_test(ast, mcts_params)
```
where ``mcts_params`` is a ``DPWParams`` object containing the Monte Carlo tree search parameters.  This method applies a heuristic that tries to push the search deeper into the time sequence.  Specifically, the algorithm will commit to the best child found after ``iterations``.  Iterations thereafter will assume that the step is fixed and root the MCTS search starting at the next time step, i.e., "online MCTS".  If you have episodes with many time steps, this may be a good heuristic to try.

The traditional MCTS method can be called using
```julia
result = stress_test2(ast, mcts_params)
```
All sampling will start at the first time step thus giving global optimization properties as the number of iterations goes to infinity.

The result object contains the total reward, action sequence, and q-values for the best k execution paths found (includes all rollouts sampled). 
```julia
result.rewards[k]
result.action_seqs[k]
result.q_values[k]
```

To replay the simulator through a sequence of seeds to recover the trajectory, use
```julia
#Optionally, first activate user logging routines in the sim here.  Then,
reward, actions = play_sequence(ast, result.actions_seqs[k]) #steps the simulator through a sequence of actions
```

For full working examples, see the Walk1D example in the ``examples`` folder. 

## References

[1] R. Lee, M. J. Kochenderfer, O. J. Mengshoel, G. P. Brat, and M. P. Owen, "Adaptive Stress Testing of Airborne Collision Avoidance Systems," in Digital Avionics Systems Conference (DASC), Prague, Czech Republic, 2015 

[![Build Status](https://travis-ci.org/sisl/AdaptiveStressTesting.jl.svg?branch=master)](https://travis-ci.org/sisl/AdaptiveStressTesting.jl)
[![Coverage Status](https://coveralls.io/repos/sisl/AdaptiveStressTesting.jl/badge.svg)](https://coveralls.io/r/sisl/AdaptiveStressTesting.jl)
