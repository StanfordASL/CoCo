using Primes

function generateHaltonSamples(dim::Int, n::Int, sf::Array=[1])
  primeNums = primes(1000);

  if length(sf) != dim
    # warn("Scale factors for Halton samples being set to 1 by default!")
    sf = ones(dim)
  end

  samples = zeros(dim,n)

  for (idx, base_val) in enumerate(primeNums[1:dim]) 
    samples[idx, :] = sf[idx] > 0 ? sf[idx]*generateHaltonSequence(n, base_val) : -2*sf[idx]*(generateHaltonSequence(n, base_val) - 0.5)
  end
  samples
end

function generateHaltonSequence(n::Int, base::Int) 
  halton_sequence = zeros(n,1)

  for idx in range(1, stop=n, step=1)
    halton_sequence[idx] = localHaltonSingleNumber(idx, Float64(base))
  end
  halton_sequence
end

function localHaltonSingleNumber(n::Int, b::Float64)
  n0 = n
  hn = 0
  f = 1/b

  while(n0 > 0)
    n1 = floor(n0/b)
    r = n0 - b*n1
    hn = hn + f*r
    f = f/b
    n0 = n1
  end
  hn
end
