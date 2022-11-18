# An example application that shows how to use custom emission functions.
# 
# NOTE: Use with the "combo" data set, eg (6 is the number of states):
#       combo.py 6 combo 
#
# The "combo" data set has two columns: one numerical and one categorical.
# The emission probability must therefore be a combination of a Poisson
# and a Histogram probability function.
#
# This example shows how to create the corresponding custom functions and
# use them with the hmmteach module.

import sys
import numpy as np

import hmmteach as hmm

def emission( x, p ):
    label, value = x
    histo, lambd = p
    
    prob1 = hmm._histogram( label, histo )
    prob2 = hmm._poisson( value, lambd )

    return prob1*prob2

def calculator( data, occ ):
    d1 = []
    d2 = np.empty( len(data) )

    for i in range(len(data)):
        lab, val = data[i]
        d1.append( lab )
        d2[i] = val

    p1 = hmm._histogram_calculator( d1, occ )
    p2 = hmm._poisson_calculator( d2, occ )

    return list( zip( p1, p2 ) )

# -----

states = int( sys.argv[1] )
data   = sys.argv[2] 

full = True

tmp = []
with open(data) as file:
    for line in file:
        lab, val = line.strip().split()
        tmp.append( ( lab, int(val) ) )
data = tmp        

model = hmm.Model( states, emission, calculator )
model.emitparams = [ ( {'in':.2, 'out':.8}, 13 ), ( {'in':.8, 'out':.2}, 13 ),
                     ( {'in':.2, 'out':.8}, 20 ), ( {'in':.8, 'out':.2}, 20 ),
                     ( {'in':.2, 'out':.8}, 30 ), ( {'in':.8, 'out':.2}, 30 ) ]

model.fit( data, display=True, full=full )
print( model )

_, path, _ = model.evaluate( data, full=full )
trs = hmm.count_transitions( states, path )
print( "Transition Count:\n", trs )
