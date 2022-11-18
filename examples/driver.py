# Example application that loads a data file, fits an HMM, and prints results.
# Command line args: number-of-states poisson|gauss|histogram datafile

import argparse
import numpy as np

import hmmteach as hmm

parser = argparse.ArgumentParser()
parser.add_argument( 'states', help="Number of states", type=int )
parser.add_argument( 'distro', choices=['poisson', 'gauss', 'histogram'] )
parser.add_argument( 'data', help="Data file", type=str )
parser.parse_args( namespace=__builtins__ )

full = True

if distro in [ 'poisson', 'gauss' ]:
    data = np.loadtxt( data )
else:
    data = np.loadtxt( data, dtype=str )

model = hmm.Model( states, distro )
model.specify( emitparams=hmm.guess_emitparams(states, data, distro) )

model.fit( data, display=True, full=full )
print( model )

_, path, _ = model.evaluate( data, full=full )
trs = hmm.count_transitions( states, path )
print( "Transition Count:\n", trs )
