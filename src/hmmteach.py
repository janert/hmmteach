"""A pure-Python implementation to fit and evaluate Hidden-Markov Models."""

import copy
import math
import numpy as np

# Data can be totally opaque.
# Only used as input to emissions and to recalc emission parameters.
#
# Emission probs must be fct of two args: fct( data, param ).
# Both args can be totally opaque. 
#
# When using custom fcts, must provide emission param calculator fct:
# calc( data, occ_prob ) -> [ params ]
#
# "emissions" can be string, sgl fct, vector of fcts
# "emitparams" is a vector of emission params
#
# Baked in emissions: gauss, poisson, histogram
# Param types are [ (mu,sigma), ... ], [ lambda, ... ], ???
#
# When using custom distrib AND wanting to generate/forecast, only then a
# signal generator must be provided as well. Takes params, returns signal value
#
#
# 1) define the model: num of states and form of emission prob
# 2) specify the model (either fit, or from elsewhere): trsf and emitparams
# 3) evaluate the model: output prob and seq of states for data set
# 4) generate by the model: forecast or generate a typical stream
#
#
# Conventions:
# m : number of states
# n : number of timesteps/observations
# s, u : states
# t : time
#
#
# ONLY __init__, specify() change the internal state of the model directly,
# fit() does so by calling specify.
# ALL other member fcts call do NOT change the internal state, and return all
# results.
#
# The "model" comprises only m, trsf, and emitparams.
# Data-related things (incl fwd, bwd, etc) are not part of the "model".
#

# ==================================================
# Helper functions for probability distribution and random numbers

# Global singleton instance of random number generator
_RNG = np.random.default_rng()

# -----

# Probability for signal value
def _poisson( x, lambd ):
    return math.pow(lambd, x)*math.exp(-lambd)/float(math.factorial(int(x)))
    
def _gauss( x, params ):
    mu, sigma = params
    return math.exp( -0.5*((x-mu)/sigma)**2 )/(math.sqrt(2.0*math.pi)*sigma)

def _histogram( x, histo ):
    return histo.get( x, 0.0 )

# -----

# Random signal generators
def _poisson_random_generator( lambd ):
    return _RNG.poisson( lambd )

def _gauss_random_generator( params ):
    mu, sigma = params
    return _RNG.normal( mu, sigma )    

def _histogram_random_generator( histo ):    
    r = _RNG.random()
    s = 0.0
    for k, v in histo.items():
        s += v
        if r <= s:
            return k
        
    raise Exception( "Never get here" )

# -----

# Estimate params of prob distributions from data
def _poisson_calculator( data, occprob ):
    # Let occprob[s][t]

    m, n = occprob.shape
    
    if type(data) != type( np.array ):
        data = np.array( data )

    res = []
    for s in range(m):
        if sum(occprob[s,:]) > 0.0:
            res.append( np.dot( data, occprob[s,:] )/sum(occprob[s,:]) )
        else:
            res.append( 0 )
            
    return res

def _gauss_calculator( data, occprob ):
    m, n = occprob.shape
    
    if type(data) != type( np.array ):
        data = np.array( data )

    res = []
    for s in range(m):
        if sum(occprob[s,:]) > 0.0:
            mu = np.average( data, weights=occprob[s,:] )
            var = np.average( (data - mu)**2, weights=occprob[s,:] )
            res.append( (mu, var) )
        else:
            res.append( (0.0, 1.0) )
        
    return res

def _histogram_calculator( data, occprob ):
    m, n = occprob.shape
    
    res = []
    for s in range(m):
        
        norm, histo = 0.0, {}        
        for t in range(n):
            if data[t] not in histo:
                histo[data[t]] = 0.0
            
            histo[ data[t] ] += occprob[s,t]
            norm += occprob[s,t]
            
        for key in histo:
            if norm > 0.0:
                histo[key] /= norm
            else:
                histo[key] = 0.0
            
        res.append( histo )
            
    return res

# ==================================================
# HMM Model

class Model:
    def __init__( self, states, emission, calculator=None, signaller=None ):
        """Define a Hidden Markov Model: number of states and emission 
           probabilities

        states: number of hidden states
        emission: "poisson", "gauss", or "histogram" to use one of the 
                  built-in distribution functions, or a function f(x,p),
                  where x is a data point and p the required parameters,
                  returning the probability for x to occur.
        calculator: a function f( data, occ ), where data is a data set 
                  of n elements, and occ is an m-x-n matrix of occupation
                  probabilities, that estimates the emission parameters
                  from its inputs, returning an array of parameters.
                  Only required when using custom emission probabilities.
        signaller: a function f( p ), with parameters p, returning a random
                  data point. Only required when using custom emission
                  probabilities.
        """
        self.m = states

        self.start = np.ones(states)/states
        self.transfer = self._transfer( states )
        
        if type(emission) != type( "" ):
            self.emission = emission
            self.emitparams = [ None for _ in range(self.m) ]
            self.calculator = calculator
            self.signalmaker = signaller

            # If list of emission probs, one per state:
            if type(emission) == type( [] ):
                raise Exception( "State-Specific Emissions Not Implemented" )
           
        elif emission == "poisson":
            self.emission = _poisson            
            self.emitparams = [ 1 for _ in range(self.m) ]
            self.calculator = _poisson_calculator
            self.signalmaker = _poisson_random_generator
            
        elif emission == "gauss":
            self.emission = _gauss
            self.emitparams = [ (0,1) for _ in range(self.m) ]
            self.calculator = _gauss_calculator
            self.signalmaker = _gauss_random_generator            
            
        elif emission == "histogram":
            self.emission = _histogram
            self.emitparams = [ {} for _ in range(self.m) ]
            self.calculator = _histogram_calculator
            self.signalmaker = _histogram_random_generator

        elif emission == "clone":
            pass
            
        else:
            raise Exception( "Unknown Emission Prob" )

    def __str__( self ):
        s = "\n" + "="*50 + "\n"
        
        s += "Transfer Matrix and Occupation Probabilities:\n"
        for i in range(self.m):
            s += "[ "
            for j in range(self.m):
                s += "%4.2f " % self.transfer[i,j]
            s += "]\t%f" % np.sum( self.transfer[i,:] )
            s += "\t[%4.2f]\n" % self.start[i]

        s += "\nEmission Parameters:\n"
        for i in range(self.m):
            s += "State %d:\t" %i + str( self.emitparams[i] ) + "\n"
            # s += pprint.pformat( self.emitparams[i] ) + "\n"            
            
        s += "="*50 + "\n" 
        return s
        
    def getmodel( self ):
        """Returns the transfer matrix, emission parameters, and 
           average state occupation probabilities."""
        return self.transfer, self.emitparams, self.start
    
    def clone( self ):
        """Returns a deep-copy clone of the current model."""
    
        model = Model( self.m, "clone" )

        np.copyto( model.transfer, self.transfer )        
        np.copyto( model.start, self.start )
        model.emitparams = copy.deepcopy( self.emitparams )

        model.emission = self.emission
        model.calculator = self.calculator
        model.signalmaker = self.signalmaker

        return model
    
    # -----
    # Public functions

    def specify( self, transfer=None, emitparams=None, start=None ):
        """Specify the model: transfer matrix, emission parameters, start 
        occupation probabilities. Omitted parameters are left untouched."""
        
        if transfer is not None:
            self.transfer = transfer
            
        if emitparams is not None:
            self.emitparams = emitparams
            
        if start is not None:
            self.start = start
    
    def fit( self, data, full=True, display=False, maxitr=50, cutoff=1.0001 ):
        """Fit the model to a data set.

        data: an iterable of n elements
        full: use Baum-Welch algorithm if True, Viterbi otherwise, 
        display: write progress info to stdout if True
        maxitr: max number of Expectation-Maximization steps
        cutoff: stop iteration if relative improvement is less than cutoff

        Returns number of iterations performed.
        """
        
        prv_ttl = None
        for itr in range( maxitr ):
            if full:
                fwd, ttl1 = self._forward( data )
                bwd, ttl2 = self._backward( data )
                trsf,emitparams,start = self._baum_welch(data,fwd,bwd,fwd*bwd)
                ttl = ttl1
                
            else:
                fwd, ttl, path = self._opa( data )
                trsf, emitparams, start = self._viterbi( data, path )

            self.specify( trsf, emitparams, start )

            improvement = ttl/prv_ttl if prv_ttl else 1e16
            prv_ttl = ttl
                
            if display:
                print( "%d %g %g" % ( itr, ttl, improvement ) )

            if improvement < cutoff:
                break

        return itr
    
    def evaluate( self, data, full=True ):
        """Evaluate the model for a data set.

        data: an iterable of n elements
        full: use Optimal Path Approximation if False

        returns: total production probability,
                 state sequence as n-element integer array of states,
                 occupation probabilities as m-x-n matrix 
        """
        m = self.m
        n = len(data)
        
        if not full:
            fwd, total, path = self._opa( data, path=True )
            occ = np.zeros( (m,n) )
            for t in range(n):
                occ[ path[t],t ] = 1
            return total, path, occ

        # If full is True:
        fwd, ttl1 = self._forward( data )
        bwd, ttl2 = self._backward( data )
        occ = self._occprob( fwd, bwd ) # occ = fwd*bwd/ttl1

        path = np.zeros( n, dtype=int )
        for t in range(n):
            path[t] = np.argmax( occ[:,t] )
        
        return ttl1, path, occ
    
    def generate( self, steps, start=None, scalarsignals=True ):
        """Generate data from the model.

        steps: number of data points to generate
        start: initial state occupation probabilities as m-element vector
        scalarsignals: use NumPy array for results if True, 
                       Python array otherwise

        returns array of generated data points, sequence of states
        """
        
        path = np.zeros( steps, dtype=int )
        signals = np.zeros(steps) if scalarsignals else [None]*steps

        crr = start if start is not None else self.start
        for t in range(steps):
            # Propagate
            crr = np.matmul( crr, self.transfer )
            
            # Select one state
            s = np.searchsorted( np.cumsum( crr ), _RNG.random() )
            path[t] = s
            
            # Create Signal
            signals[t] = self.signalmaker( self.emitparams[s] )

            # Create new occ vector
            crr = np.zeros( self.transfer.shape[1] )
            crr[s] = 1.0
            
        return signals, path
   
    # -----
    # Setup functions
    
    def _transfer( self, m ):
        if m == 1:
            return np.eye(m)
        
        diagonal = 0.9
        offdiag = (1.0 - diagonal)/(m-1)
        
        return (diagonal-offdiag)*np.eye(m) + offdiag*np.ones( (m,m) )
        
    # -----
    # Main worker functions
    
    def _forward( self, data ):
        m = self.m
        n = len(data)
        
        fwd = np.zeros( (m,n) )

        for s in range(m):
            fwd[s,0] = self.start[s]
            fwd[s,0] *= self.emission( data[0], self.emitparams[s] )
            
        for t in range( 1, n ):
            for s in range(m):
                fwd[s,t] = np.dot( self.transfer[:,s], fwd[:,t-1] )
                fwd[s,t] *= self.emission( data[t], self.emitparams[s] )

        total = np.sum( fwd[:,n-1] )     # Sum over last col
       
        return fwd, total

    
    def _opa( self, data, path=True ):
        m = self.m
        n = len(data)
        
        fwd = np.zeros( (m,n) )
        psi = np.zeros( (m,n), dtype=int )
       
        for s in range(m):
            fwd[s,0] = self.start[s]
            fwd[s,0] *= self.emission( data[0], self.emitparams[s] )
            psi[s,0] = int(np.argmax(self.start))
            
        for t in range( 1, n ):
            for s in range(m):
                fwd[s,t] = np.max( self.transfer[s,:] * fwd[:,t-1] )
                fwd[s,t] *= self.emission( data[t], self.emitparams[s] )
                psi[s,t] = int(np.argmax( self.transfer[s,:] * fwd[:,t-1] ))
                
        total = np.max( fwd[:,n-1] )      # Max of last col

        if not path:
            return fwd, total
        
        path = self._backtrack( fwd, psi )
        return fwd, total, path

    
    def _backtrack( self, fwd, psi ):
        m, n = psi.shape
        
        res = np.zeros( n, dtype=int )
        res[n-1] = int(np.argmax( fwd[:,n-1] ))
        
        for i in range( n-2, -1, -1 ):
            res[i] = psi[ res[i+1], i+1 ]
            
        return res

    
    def _backward( self, data ):
        m = self.m
        n = len(data)
        
        bwd = np.zeros( (m,n) )

        for s in range(m):
            bwd[s,n-1] = 1.0

        for t in range( n-2, -1, -1 ):
            for s in range(m):
                for u in range(m):
                    tmp = self.transfer[s,u]
                    tmp *= self.emission( data[t+1], self.emitparams[u] )
                    tmp *= bwd[u, t+1]
                    bwd[s,t] += tmp

        total = 0
        for s in range(m):
            tmp = self.emission( data[0], self.emitparams[s] )
            total += bwd[s,0]*self.start[s]*tmp
            
        return bwd, total        

    
    def _viterbi( self, data, path ):
        m = self.m
        n = len(path)

        # --- Transition
        occ = np.zeros( m )
        trs = np.zeros( (m,m) )
        
        for t in range(1, n):
            prv, crr = path[t-1], path[t]
            
            occ[ prv ] += 1.0
            trs[ prv, crr ] += 1.0

        for s in range(m):
            if occ[s] == 0.0:
                continue
            
            for u in range(m):
                trs[s,u] /= occ[s]    # path[n] not incl, by construction
                
        # --- Emission
        occprob = np.zeros( (m,n) )
        for t in range(n):
            occprob[ path[t], t ] = 1.0

        emitparams = self.calculator( data, occprob )

        # --- Dwell
        dwell = np.zeros(m)
        for s in range(m):
            dwell[s] = np.count_nonzero( path == s )

        return trs, emitparams, dwell/n

    
    def _occprob( self, forward, backward ):
        m, n = forward.shape
        total = np.sum( forward[:,n-1] )  # Sum over last col
        return forward*backward/total
                

    def _baum_welch( self, data, fwd, bwd, occ ):
        m, n = occ.shape
            
        # --- Transition
        trs = np.zeros( (m,m) )
        for s in range(m):
            norm = np.sum( occ[s,:n-1] )  # Sum over row, omitting last col!
            if norm == 0.0:
                continue
            
            for u in range(m):
                for t in range(1,n):
                    tmp = fwd[s,t-1]
                    tmp *= self.transfer[s,u]
                    tmp *= self.emission(data[t], self.emitparams[u])
                    tmp *= bwd[u,t]
                    trs[s,u] += tmp
                    
                trs[s,u] /= norm
        
        # --- Emission
        emitparams = self.calculator( data, occ )

        # --- Dwell
        dwell = np.zeros(m)
        for s in range(m):
            dwell[s] = np.sum( occ[s,:] ) # Sum over row, incl last col!
        dwell /= n*np.sum( fwd[:,n-1] )   # Sum over last col: total prob
        
        return trs, emitparams, dwell

    
# ==================================================
# Convenience functions

# Count actual transitions in path
def count_transitions( states, path ):
    """Returns a matrix with the integer count of state transitions in
    provided sequence of states."""
    
    trs = np.zeros( (states,states), dtype=int )
    
    for t in range(1, path.size):
        prv, crr = path[t-1], path[t]
        trs[ prv, crr ] += 1

    return trs


# Guess likely starting values for emitparams from data. (Very ad-hoc.)
def guess_emitparams( states, data, emission ):
    """Given a number of states, a data set, and an emission family, 
    returns an array of emission parameters, one for each state. 
    The emission family must be one of "histogram", "poisson", or "gauss".
    """
    
    emitparams = []
        
    if emission == "histogram":
        # histogram of all data points, in order to find all possible values
        hs = _histogram_calculator( data, np.ones( (1,len(data)) ) )
        keys = hs[0].keys()   # observed signal values
            
        for s in range(states):
            histo = {}
            r = _RNG.random( len(keys) )
            for i, k in enumerate(keys):
                histo[k] = r[i]/sum(r)
                    
            emitparams.append( histo )
                
        return emitparams
                
    if emission in [ "poisson", "gauss" ]:
        if type(data) != type( np.array ):
            data = np.array( data )

        q = 0
        for s in range(states):
            q += 100/(states+1)
            p = np.percentile( data, q )

            if emission == "poisson":
                emitparams.append( p )

            elif emission == "gauss":
                d = np.std(data)/states
                emitparams.append( (p,d) )

        return emitparams

    raise Exception("Unknown emission %s in _guess_emitparams" % emission)
                
# ==================================================

def all_in_one( states, emission, data, full=True ): 
    """Takes the number of states, an emission family, and the name of a 
    data file. Fits the model, and prints the results to stdout. Uses
    Baum-Welch algorithm if full is True, Viterbi otherwise. The 
    emission family must be one of "histogram", "poisson", or "gauss"."""
    
    data = np.loadtxt( data )

    model = Model( states, emission )
    model.specify( emitparams=guess_emitparams(states, data, emission) )

    model.fit( data, display=True, full=full )
    print( model )

    _, path, _ = model.evaluate( data, full=full )
    trs = count_transitions( states, path )
    print( "Transition Count:\n", trs )
    
    
def bootstrap( states, emission, data, samples=10, full=True ):
    """Takes the number of states, an emission family, and the name of a 
    data file. Fits the model, and then performs samples parametric
    bootstrap steps. Prints the results for the transfer matrix and
    emission parameters to stdout. Uses Baum-Welch algorithm if full 
    is True, Viterbi otherwise. The emission family must be one of 
    "histogram", "poisson", or "gauss"."""
    
    data = np.loadtxt( data )

    # Create and fit model
    model = Model( states, emission )               
    model.specify( emitparams=guess_emitparams(states, data, emission) )
    model.fit( data, display=False, full=full )

    # Containers for generated transfer matrices and emitparams structures
    ts = np.empty( (samples, states, states) )
    ps = []
    for k in range(samples):
        # Create bootstrap sample: "signals"
        signals, path = model.generate( len(data) )

        # Create and fit model to bootstrap sample
        m2 = model.clone()
        m2.fit( signals, display=False, full=full )
        # print( model )

        # Store results
        trsf, emits, _ = m2.getmodel()
        ts[k] = trsf
        ps.append( emits )

    # Average and print the bootstrap samples: for transfer matrix ...
    avg, stdvar = np.mean(ts, axis=0), np.std(ts, axis=0)
    for m1 in range(states):
        for m2 in range(states):
            print( "%.3f +/- %.3f  " % ( avg[m1,m2], stdvar[m1,m2] ), end="" )
        print( "" )        
    print( "" )
        
    # ... and for emission parameters
    keys, avg, stdvar = _summarize_bootstrap_samples(ps, states, emission, data)
    for m in range( states ):
        for d, k in enumerate(keys):
            print( "%s: %.3f +/- %.3f " % ( keys[d], avg[m,d], stdvar[m,d] ) )
        print( "" )
        
# Average emission parameters over all bootstrap samples. This is painful,
# because the structure of the emission parameters is different for each
# type of distribution.
def _summarize_bootstrap_samples( ps, states, emission, data ):
    
    keys = [ 'lambda' ] if emission == "poisson" else [ 'mu', 'sigma' ]
    if emission == "histogram":
        # histogram of all data points, in order to find all possible values
        keys = _histogram_calculator(data, np.ones( (1,data.size) ))
        keys = list( keys[0].keys() )

    # depth is the number of parameters per state
    depth = { 'poisson': 1, 'gauss': 2, 'histogram': len(keys) }
    
    # Function to pick out a single parameter, based on "emission"
    def access( p ):
        if emission == "poisson": return p
        elif emission == "gauss": return p[d]
        elif emission == "histogram": return p.get( keys[d], 0 )
    
    # Reshape ps into a three-dim cube with dimension n x m x depth
    cube = np.empty( (len(ps), states, depth[emission]) )    
    for t in range(len(ps)):
        for m in range(states):
            for d in range( depth[emission] ):
                cube[t, m, d] = access( ps[t][m] )
                
    avg = np.mean( cube, axis=0 )
    stdvar = np.std( cube, axis=0 )
    
    return keys, avg, stdvar


if __name__ == "__main__":
#    all_in_one( 3, "poisson", "poisson3", full=True )
    bootstrap( 3, "poisson", "poisson3", full=True, samples=20 )
