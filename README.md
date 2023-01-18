
# hmmteach

A pure-Python implementation to fit and evaluate Hidden-Markov Models.


**NOTE:**
This module is not recommended for production use on large data sets.
Consider the [hmmlearn](https://github.com/hmmlearn/hmmlearn) library instead.


## Description:

This module provides pure Python implementations of the basic algorithms
to work with Hidden-Markov Models (HMM). 

The primary intent was to provide straightforward implementations of the
required algorithms, to aid understanding and experimentation. Simplicity
and transparency of the implementation therefore took priority. Little 
attempt has been made to optimize the runtime performance of the code, 
and none to deal with floating-point underflow problems. 

The API has been kept simple and practical. The interface is meant to
clearly outline and reflect the different tasks or "problems" posed
by HMMs. 

HMM algorithms are computationally intensive. Moreover, they involve the
repeated multiplication of probabilities, quickly leading to problems of
floating-point underflow. Special measures need to be taken to prevent
them for larger data sets, but this necessarily obscures the algorithms. 

Acknowledging these limitations, the present implementation is usable for 
small data sets up to a few hundred points. For larger data sets, consider
the [hmmlearn](https://github.com/hmmlearn/hmmlearn) library, which moves 
the core computation to C++ to improve performance, and also handles the 
underflow problem via logarithmic representation.


## Overview:

### The Model

An HMM is fully defined by three quantities:
- the number of hidden states _m_
- the transition matrix among states (an _m x m_ matrix)
- a set of _m_ emission probabilities and the parameters they require

By contrast, any data set or any calculated quantity directly derived 
from a data set is _not_ part of the model, and is not stored in a
model instance.

The `Model` class encapsulates an HMM. It provides a set of "public"
functions to perform the following activities:

1. Define the model
2. Specify the model
3. Train the model on a data set
4. Evaluate the model for a data set
5. Generate a data set from the model

The class also provides several "non-public" methods to do the actual work.


### Emission Probabilities

The HMM concept is independent of the type and nature of the observation
data: the data can be totally opaque. The present implementation makes no 
assumptions about the data, and never tries to read it. It only assumes
that data is supplied as a collection that is indexable (an iterable).

The HMM concept also makes no assumptions about the specific nature and 
structure of the emission probability distributions. For convenience, the 
module provides implementations for three common distributions: empirical
**Histograms** (for categorical data), **Poisson** (for low integer counts), 
and **Gaussian** (for univariate numerical data). Multi-variate Gaussian and 
Gaussian Mixture distributions may be added in the future.

It is also possible to provide custom distribution functions to describe
data sets not represented adequately by the built-in distributions. Three
functions are required:

- A function that returns the probability for a certain data point `x` to 
  occur: `prob( x, params ) -> float`. (Always required.)
- A function to calculate the parameter values, given a data set and a 
  matrix that gives, for each data point, the occupation probability of
  each state: `calc( dataset, occupation ) -> [ params, ... ]`. Here,
  `dataset` is an iterable of `n` elements, `occupation` is an `m x n`
  NumPy matrix, and the return value is an array of `m` elements, each
  element represents the emission parameters for one state. (Only
  required when training a model.)
- A random generator that returns a valid data point, given a set of
  parameters: `rand( params ) -> x`, where `x` is a valid data point.
  (Only required with generating data points from a model.)

Like the data points, the parameters required by each probability 
distribution are opaque to this module. They may be scalar, tuples,
dictionaries, or arbitrary objects. The module does not read or interpret
them, but it will make sure to supply the correct parameter value for the 
current state to the custom function when required. It is up to the custom 
functions to parse and interpret the parameter values passed into them 
appropriately.


### Convenience Functions

This module provides built-in support for empirical histograms, Poisson, 
and univariate Gaussian distributions. It also includes a function to
calculate plausible emission parameters from a data set, one for each
state. These can be used as starting values when training a model.

Also included are two entry-point functions, to train, evaluate, and
verify a model (verification is done via parametric bootstrap). 

Treating both the data and the emission parameters as opaque involves
a trade-off: it keeps the current implementation perfectly general,
and also emphasizes the boundaries of the HMM concept most clearly. 
At the same time, it limits the convenience functions that this
implementation can provide.


## Reference:

`class hmmteach.Model( states, emission, calculator=None, signaller=None )`

Defines a Hidden Markov Model: number of states and emission probabilities.

- states: number of hidden states
- emission: "poisson", "gauss", or "histogram" to use one of the 
                  built-in distribution functions, or a function f(x,p),
                  where x is a data point and p the required parameters,
                  returning the probability for x to occur.
- calculator: a function f( data, occ ), where data is a data set 
                  of n elements, and occ is an m-x-n matrix of occupation
                  probabilities, that estimates the emission parameters
                  from its inputs, returning an array of parameters.
                  Only required when using custom emission probabilities.
- signaller: a function f( p ), with parameters p, returning a random
                  data point. Only required when using custom emission
                  probabilities.

`model.getmodel()`

Returns the transfer matrix, emission parameters, and average state 
occupation probabilities.
      
`model.clone()`

Returns a deep-copy clone of the current model.

`model.specify( transfer=None, emitparams=None, start=None )`

Specifies the model: transfer matrix, emission parameters, start 
occupation probabilities. Omitted parameters are left untouched.

`model.fit( data, full=True, display=False, maxitr=50, cutoff=1.0001 )`

Fits the model to a data set.

- data: an iterable of n elements
- full: use Baum-Welch algorithm if True, Viterbi otherwise, 
- display: write progress info to stdout if True
- maxitr: max number of Expectation-Maximization steps
- cutoff: stop iteration if relative improvement is less than cutoff

Returns number of iterations performed.

`model.evaluate( data, full=True )`

Evaluates the model for a data set.

- data: an iterable of n elements
- full: use Optimal Path Approximation if False

Returns: ( total production probability,
           state sequence as n-element integer array of states,
           occupation probabilities as m-x-n matrix )

`model.generate( steps, start=None, scalarsignals=True )`

Generates data from the model.

- steps: number of data points to generate
- start: initial state occupation probabilities as m-element vector
- scalarsignals: use NumPy array for results if True, 
               Python array otherwise

Returns: ( array of generated data points, sequence of states )


`hmmteach.all_in_one( states, emission, data, full=True )`

Takes the number of states, an emission family, and the name of a 
data file. Fits the model, and prints the results to stdout. Uses
Baum-Welch algorithm if full is True, Viterbi otherwise. The 
emission family must be one of "histogram", "poisson", or "gauss".
    

`hmmteach.bootstrap( states, emission, data, samples=10, full=True )`

Takes the number of states, an emission family, and the name of a 
data file. Fits the model, and then performs samples parametric
bootstrap steps. Prints the results for the transfer matrix and
emission parameters to stdout. Uses Baum-Welch algorithm if full 
is True, Viterbi otherwise. The emission family must be one of 
"histogram", "poisson", or "gauss".


## Example:

```python
import numpy as np
import hmmteach as hmm

states = 3
distro = "poisson"
data = np.loadtxt( "dataset.txt" )
full = True

model = hmm.Model( states, distro )
model.specify( emitparams=hmm.guess_emitparams(states, data, distro) )

model.fit( data, display=True, full=full )
print( model )

_, path, _ = model.evaluate( data, full=full )
trs = hmm.count_transitions( states, path )
print( "Transition Count:\n", trs )
```

The `examples` directory contains additional example uses; the `data` 
directory some sample data sets. The names of the data sets suggest 
suitable emission distributions and a plausible guess for the number 
of states. See the comments at the start of the source files for details.


## Dependencies and Installation:

This module requires NumPy.

For now, simply drop `hmmteach.py` somewhere where Python can find it,
adjusting the `PYTHONPATH` if necessary. 


## Issues:

When training a model on a data set consisting of only a single time step,
then the entries in the transition matrix are 0, not 1. (That's because
the transition matrix is calculated based on the number of inferred state
transitions in the data set, and for a data set of _n=1_, there are not
such transitions.)


## License:

Copyright (c) 2022, Philipp K. Janert. 
All rights reserved.

Unless otherwise stated in individual files, the contents of this 
repository is licensed under the BSD 3-Clause license found in the
LICENSE file in this directory. 
