function [kernel] = kernel_quasiperiodic(X,y, opt)
% 	kernel_quasiperiodic(opt)
%	Computes the kernel matrix for a univariate quasiperiodic kernel, 
%   convex sum of periodic and gaussian kernel.
%	INPUTS:
%		-OPT: struct with the following options:
%       - paramsel : struct containing the following fields (computed by paramsel_*).
%		- sigma : width of the gaussian kernel.
%       -X: input data matrix
%	
%	OUTPUT: struct with the following fields:
%		-type: 'quasiperiodic'
%		-K: kernel matrix

kernel = opt.kernel;
kernel.type = 'quasiperiodic';

if ~isfield(kernel,'distance')
	kernel.distance = square_distance(X',X');
end	

if ~isfield(kernel, 'kerrange')
    n = size(kernel.distance,1);
    if ~isprop(opt,'sigmamin')
        D = sort(kernel.distance(tril(true(n),-1)));
        firstPercentile = round(0.01*numel(D)+0.5);
        opt.newprop('sigmamin', sqrt(D(firstPercentile)));
        clear D;
    end
    if ~isprop(opt,'sigmamax')
        opt.newprop('sigmamax', sqrt(max(max(kernel.distance))));
    end
    if opt.sigmamin <= 0
        opt.sigmamin = eps;
    end
    if opt.sigmamin <= 0
        opt.sigmamax = eps;
    end	
    q = (opt.sigmamax/opt.sigmamin)^(1/(opt.nsigma-1));
    kernel.kerrange = opt.sigmamin*(q.^(opt.nsigma:-1:0));
end

if ~isfield(kernel, 'init') || ~kernel.init
    
    if ~isfield(opt.paramsel, 'sigma')
        sigma = kernel.kerrange(opt.paramsel.sigmanum);
    else
        sigma = opt.paramsel.sigma;
    end
    
    D = kernel.distance;
    K = opt.paramsel.alpha*exp(-sin(((D.^(1/2)).*(pi/opt.period))).^2);
    K = K+ (1-opt.paramsel.alpha)*exp(-D./(sigma^2));
    kernel.K = K;
else
    kernel.init = 0;
end

