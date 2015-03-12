function [A b] = gen_data_2(n, d, prob1, scale, prob, mag)

  if nargin < 6
      mag = 500;
  end

  if nargin < 5
      prob = 0.001;
  end

  if nargin < 4
      scale = 0.2;
  end

  if nargin < 3
      prob1 = 0.8;
  end

  % begin binary seraching the acceptable nonzero number of the first column
  f = @(x) 1 - (1 - 0.01)^x - prob1;

  aa = 1;
  bb = n/d;

  while abs(bb - aa) > 1
    cc = (aa + bb)/2;
    if f(cc) > 0
      bb = cc;
    else
      aa = cc;
    end
  end

  init = ceil(bb);
  

  % begin binary searching the optimal ratio
  g = @(x) init*(1 - x^d)/(1 - x) - n;  

  aa = 1;
  bb = 2;

  if g(bb) < 0
    q = 2;
    init = floor((n/(2^d-1))+0.5);
  else  
    while abs(bb - aa) > 0.001
      cc = (aa + bb)/2;
      if g(cc) > 0
        bb = cc;
      else
        aa = cc;
      end
    end

    q = aa;
  end
  
  init
  q

  cur = 1;

  row = [];
  col = [];

  for j = 1:d-1
      inc = round(init*q^(j-1));
      row = [row cur:cur+inc-1];
      col = [col j*ones(1, inc)];
      cur = cur + inc;
  end
  
  cur

  row = [row cur:n];
  col = [col d*ones(1, n-cur+1)];

  A = sparse(row, col, ones(n,1));
  A = full(A);

  x_exact = randn(d, 1);

  b_exact = A*x_exact;
  err     = laprnd([n, 1]);
  err     = scale * norm(b_exact)/norm(err) * err;
  b       = b_exact + err;
  ii      = rand(n, 1) < prob;
  b(ii)   = mag*err(ii);


