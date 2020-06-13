import numpy as np

def generate_first_N_primes(N): 
    ii, jj, flag = 0, 0, 0
    primes = []
  
    # Traverse each number from 1 to N 
    # with the help of for loop 
    for ii in range(1, N + 1, 1): 
        if (ii == 1 or ii == 0):
            # Skip 0 and 1
            continue

        # flag variable to tell 
        # if i is prime or not 
        flag = 1; 
  
        for jj in range(2, ((ii // 2) + 1), 1): 
            if (ii % jj == 0): 
                flag = 0; 
                break; 

        # flag = 1 means ii is prime 
        # and flag = 0 means ii is not prime
        if (flag == 1):
          primes.append(ii)
    return primes

def generate_halton_samples(dim, n, sf=[1]):
  primeNums = generate_first_N_primes(1000)

  if len(sf) is not dim:
    sf = np.ones(dim) 

  samples = np.zeros((dim,n))

  for idx, base_val in enumerate(primeNums[:dim]):
    if sf[idx] > 0:
      samples[idx,:] = sf[idx]*generate_halton_sequence(n, base_val)
    else:
      samples[idx,:] = -2*sf[idx]*(generate_halton_sequence(n, base_val) - 0.5)
  return samples

def generate_halton_sequence(n, base):
  halton_sequence = np.zeros(n)

  for idx, val in enumerate(range(1, n+1)):
    halton_sequence[idx] = local_halton_single_number(val, base) 

  return halton_sequence

def local_halton_single_number(n, b):
  n0 = n
  hn = 0
  f = 1/float(b)

  while n0 > 0:
    n1 = np.floor(n0/b)
    r = n0 - b*n1
    hn = hn + f*r
    f = f/float(b)
    n0 = n1

  return hn
