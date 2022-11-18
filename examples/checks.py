# Collection of test cases and driver
#
# When run without command line arguments, runs several test cases on
# short, synthetic data sets, and prints all intermediate results to
# stdout in a formatted display that aides visual inspection and
# verification.
#
# When run with command line arguments, it performs all required steps,
# using Poisson emission distributions, again while creating extensive,
# formatted output.
#     checks.py <states> <filename>

import sys
import numpy as np

import hmmteach as hmm

# ==================================================
# Formatted output: development aid
    
def print_vector( vec ):
    if type(vec) == type( [] ):
        fmt = "%4s "    # This assumes signal is str, not general obj!
    else:
        fmt = "%4.2f " if vec.dtype != np.dtype("int") else "%4d "

    print( "   ( ", end="" )
    for v in vec:
        print( fmt % v, end="" )
    print( ")" )

def print_tableau( tbl, data ):
    m, n = tbl.shape

    fmt = "%4.2f " if type(data[0]) != np.dtype("str_") else "%4s "
    if data is not None:
        print( "     ", end="" )
        for t in range(n):
            print( fmt % data[t], end="" )
        print( "" )

    fmt = "%4.2f " if tbl.dtype != np.dtype("int") else "%4d "        
    for s in range(m):
        print( "%2d [ " % s, end="" )
        for t in range(n):
            if tbl[s][t] == 0:
                print( "  _  ", end="" )
            else:
                print( fmt % tbl[s,t], end="" )                
        print( "]" )

    print( "     ", end="" )
    for t in range(n):
        print( "%4d " % t, end="" )
    print( "" )
        
def print_transfer( tbl ):
    m, _ = tbl.shape

    for i in range(m):
        print( "[ ", end="" )
        for j in range(m):
            print( "%4.2f " % tbl[i,j], end="" )
        print( "]\t%f" % np.sum( tbl[i,:] ) )

def print_model( trs, emitparams, start ):
    print( "\n", "="*50, sep="" )
    print( "Transfer Matrix:" )
    print_transfer( trs )
    
    print( "\nEmission Parameters:" )
    print( emitparams )

    print( "\nOccupation Probabilities:" )
    print_vector( start )
    print( "="*50, "\n" )

# ==================================================
# Test cases for visual inspection

def run_and_print( model, data, tag="", generate=False, scalar=True ):
    fwd, ttl1 = model._forward( data )
    opa, ttlx, path = model._opa( data )
    bwd, ttl2 = model._backward( data )

    occ = model._occprob( fwd, bwd )
    
    tr1, params1, start1 = model._viterbi( data, path )
    tr2, params2, start2 = model._baum_welch( data, fwd, bwd, fwd*bwd )

    print( "\n===== %s =====" % tag )
    
    print( "\nForward: %g" % ttl1 )
    print_tableau( fwd, data )
    
    print( "\nBackward: %g" % ttl2 )
    print_tableau( bwd, data )

    print( "\nApproxi: %g" % ttlx )
    print_tableau( opa, data )
    print( "Path:" )
    print_vector( path )

    print( "\nOcc:" )
    print_tableau( occ, data )

    print( "\nViterbi:" )
    print_transfer( tr1 )
    print( "Start:", start1, "\t", np.sum(start1) )
    print( "Emits:", params1 )

    print( "\nBaum-Welch:" )
    print_transfer( tr2 )
    print( "Start:", start2, "\t", np.sum(start2) )
    print( "Emits:", params2 )

    if generate is True:
        print( "\nGenerate:" )
        signals, path = model.generate( 20, scalarsignals=scalar )
        print_vector( signals )
        print_vector( path )
            
    print( "" )
    

def test_cases():
    RNG = np.random.default_rng()

    # -----
    
    data = np.array( [1] )
    
    model = hmm.Model( 1, "poisson" )
    model.emitparams = [ 1 ]
    run_and_print( model, data, "p1x1" )

    model = hmm.Model( 1, "gauss" )
    model.emitparams = [ (1.0, 0.1) ]
    run_and_print( model, data, "g1x1" )

    model = hmm.Model( 1, "histogram" )
    model.emitparams = [ { 1: 1 } ]
    run_and_print( model, data, "h1x1" )

    # -----

    data = np.array( [1, 1] )
    
    model = hmm.Model( 1, "poisson" )
    model.emitparams = [ 1 ]
    run_and_print( model, data, "p2x1" )

    model = hmm.Model( 1, "gauss" )
    model.emitparams = [ (1.0, 0.1) ]
    run_and_print( model, data, "g2x1" )

    model = hmm.Model( 1, "histogram" )
    model.emitparams = [ { 1: 1 } ]
    run_and_print( model, data, "h2x1" )

    # -----
    
    data = np.array( [1, 1] )
    
    model = hmm.Model( 2, "poisson" )
    model.emitparams = [ 1, 5 ]
    run_and_print( model, data, "p2x2-" )

    model = hmm.Model( 2, "gauss" )
    model.emitparams = [ (1.0, 0.1), (5.0, 0.1) ]
    run_and_print( model, data, "g2x2-" )

    model = hmm.Model( 2, "histogram" )
    model.emitparams = [ { 1: 1 }, { 1: 1 } ]
    run_and_print( model, data, "h2x2-" )
    
    # -----
    
    data = np.array( [1, 5] )
    
    model = hmm.Model( 2, "poisson" )
    model.emitparams = [ 1, 5 ]
    run_and_print( model, data, "p2x2+" )

    model = hmm.Model( 2, "gauss" )
    model.emitparams = [ (1.0, 0.1), (5.0, 0.1) ]
    run_and_print( model, data, "g2x2+" )

    model = hmm.Model( 2, "histogram" )
    model.emitparams = [ { 1: 1.0, 5: 0.0 }, { 1: 0.0, 5: 1.0 } ]
    run_and_print( model, data, "h2x2+" )
    
    # -----

    data = np.hstack( (np.ones(5), 5*np.ones(5)) )
    
    model = hmm.Model( 2, "poisson" )
    model.emitparams = [ 1, 5 ]
    run_and_print( model, data, "p10x2+" )

    model = hmm.Model( 2, "gauss" )
    model.emitparams = [ (1.0, 0.1), (5.0, 0.1) ]
    run_and_print( model, data, "g10x2+" )

    model = hmm.Model( 2, "histogram" )
    model.emitparams = [ { 1: 1.0, 5: 0.0 }, { 1: 0.0, 5: 1.0 } ]
    run_and_print( model, data, "h10x2+" )

    # -----

    data = np.hstack( (np.ones(7), 5*np.ones(3), np.ones(6), 5*np.ones(4)) )
    
    model = hmm.Model( 2, "poisson" )
    model.emitparams = [ 1, 5 ]
    run_and_print( model, data, "p20x2+", True )

    model = hmm.Model( 2, "gauss" )
    model.emitparams = [ (1.0, 0.1), (5.0, 0.1) ]
    run_and_print( model, data, "g20x2+", True )

    model = hmm.Model( 2, "histogram" )
    model.emitparams = [ { 1: 1.0, 5: 0.0 }, { 1: 0.0, 5: 1.0 } ]
    run_and_print( model, data, "h20x2+", True )

    # -----

    data = np.hstack( (np.ones(6), 5*np.ones(5), np.ones(4), 5*np.ones(5)) )

    # rng = np.random.default_rng()
    
    model = hmm.Model( 2, "poisson" )
    model.emitparams = [ 1, 5 ]
    run_and_print( model, data+RNG.integers(-1,2,20), "p20x2+r", True )

    model = hmm.Model( 2, "gauss" )
    model.emitparams = [ (1.0, 0.1), (5, 0.1) ]
    run_and_print( model, data+RNG.normal(0,0.2,20), "g20x2+r", True )

    data = np.hstack( (RNG.choice( ['A', 'A', 'A', 'B', 'C'], 15 ),
                       RNG.choice( [ 'B', 'C'], 15 )) )

    model = hmm.Model( 2, "histogram" )
    model.emitparams = [ { 'A': 0.6, 'B': 0.2, 'C': 0.2 },
                         { 'A': 0.2, 'B': 0.3, 'C': 0.5 } ]
    run_and_print( model, data, "h20x2+r", True, False )

    # -----

    data = np.hstack( (np.ones(6), 5*np.ones(5), np.ones(4), 5*np.ones(5)) )
    
    model = hmm.Model( 2, "poisson" )
    model.emitparams = hmm.guess_emitparams( 2, data, "poisson" )
    run_and_print( model, data+RNG.integers(-1,2,20), "p20x2+ra", True )

    model = hmm.Model( 2, "gauss" )
    model.emitparams = hmm.guess_emitparams( 2, data, "gauss" )    
    run_and_print( model, data+RNG.normal(0,0.2,20), "g20x2+ra", True )

    data = np.hstack( (RNG.choice( ['A', 'A', 'A', 'B', 'C'], 15 ),
                       RNG.choice( [ 'B', 'C'], 15 )) )

    model = hmm.Model( 2, "histogram" )
    model.emitparams = hmm.guess_emitparams( 2, data, "histogram" )
    run_and_print( model, data, "h20x2+ra", True, False )
    
# ==================================================

def poisson( states, filename ):
    RNG = np.random.default_rng()

    states = int( states )
    data = np.loadtxt( filename )

    # Define model and starting points
    model = hmm.Model( states, "poisson" )
    model.emitparams = hmm.guess_emitparams( states, data, "poisson" )    
#    model.emitparams = [ 12, 25 ]
#    model.emitparams = [ 12, 20, 30 ]

    print( model.emitparams, "\n" )
    
    # Fit model and display result
    model.fit( data, display=True, full=True )
    print( model )

    # Randomize the input and run through fitted model
    ttl, path, occ = model.evaluate( data + RNG.integers(-1,2,len(data)) )
    print_vector( path )
    print_tableau( occ, data )
    print( ttl, "\n" )

    # Forecast 
    signals, path = model.generate( 20 )
    print_vector( signals )
    print_vector( path )

# ==================================================

# No cmdline args: run tests; otherwise, run poisson()
if len(sys.argv) == 1:
    test_cases()
else:
    poisson( *sys.argv[1:] )
