function vout = paramsel_siglamhogpregr(X,y, opt)
% paramsel_siglamhogpregr(X,y, OPT)
% Performs parameter selection for gaussian process regression.
% The hold-out approach is used.
% It selects both the noise level lambda and the kernel parameter sigma.
%
% INPUTS:
% -OPT: struct of options with the following fields:
%   fields with default values set through the defopt function:
%		- kernel.type
%		- nlambda
%               - hoperf
%
%   For more information on standard OPT fields
%   see also defopt
% 
% OUTPUT: structure with the following fields:
% -lambdas: value of the regularization parameter lambda
%           minimizing the validation error, replicated in a TX1 array 
%           where T is the number of classes
% -sigma: value of the kernel parameter minimizing the validation error
savevars = [];

if isprop(opt,'paramsel')
	vout = opt.paramsel; % lets not overwrite existing parameters.
			      		 % unless they have the same name
else
    opt.newprop('paramsel', struct());
end

[~,T]  = size(y);

if ~isprop(opt,'kernel')
    opt.newprop('kernel', struct());
	opt.kernel.type = 'rbf';
end

kerfun = str2func(['kernel_' opt.kernel.type]);
opt.kernel.init = 1;
opt.kernel = kerfun(X,y,opt);
nsigma = numel(opt.kernel.kerrange);

for i = 1:nsigma
	opt.paramsel.sigmanum = i;
	opt.kernel = kerfun(X,[],opt);
    paramsel = paramsel_hogpregr(X,y,opt);
    nh = numel(paramsel.perf);
	PERF(i,:,:) = reshape(median(reshape(cell2mat(paramsel.perf')',opt.nlambda*T,nh),2),T,opt.nlambda)';
	guesses(i,:) = median(cell2mat(paramsel.guesses'),1);
end
% The lambda axis is redefined each time but
% it is the same for all classes as it depends
% only on K so we can still sum and minimize.
%
% We have to be a bit careful when minimizing.
%
% TODO: select a lambda for each class fixing sigma.

vout.sigmas = opt.kernel.kerrange;
vout.lambda_guesses = guesses;

M = sum(PERF,3); % sum over classes
[dummy,i] = max(M(:));
[m,n] = ind2sub(size(M),i);
vout.perf = M;
vout.sigma = opt.kernel.kerrange(m);
vout.lambdas = guesses(m,n)*ones(1,T);

if numel(savevars) > 0
	[ST,I] = dbstack();
	save(ST(1).name,savevars{:});
end	
