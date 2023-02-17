% See D. L. Donoho and M. Gavish, "The Optimal Hard Threshold for Singular
% Values is 4/sqrt(3)", http://arxiv.org/abs/1305.5870
%
% This script generate Figure 1
%
% -----------------------------------------------------------------------------
% Authors: Matan Gavish and David Donoho <lastname>@stanford.edu, 2013
% 
% This program is free software: you can redistribute it and/or modify it under
% the terms of the GNU General Public License as published by the Free Software
% Foundation, either version 3 of the License, or (at your option) any later
% version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
% details.
%
% You should have received a copy of the GNU General Public License along with
% this program.  If not, see <http://www.gnu.org/licenses/>.
% -----------------------------------------------------------------------------


function figure7
 
    rng default; rng(1984);

    dists = {'gaussian','bernoulli','uniform','student-t(6)'};
    dims = [20 100];
    
    beta = 1;
    w = (8 * beta) ./ (beta + 1 + sqrt(beta.^2 + 14 * beta +1)); 
    lambdastar = sqrt(2 * (beta + 1) + w);
    
    shrink{1} = @(x)( x ); %max(x, 1+ sqrt(beta)));       %trunc WRONG
    shrink{2} = @(x)(x .* (x>=lambdastar)); %hard
    cut= [1 0];
    
    names={'$\hat{X}_r$','$\hat{X}_{\lambda_*}$'};
    colors = {'r','b'};
    
    h_dim_dist = figure('Position',[100,100,1300,600]);
    counter =1;
    for i=1:2
        for j=1:4
            fprintf('making plot %d/%d\n',counter,8);
            subplot(2,4,counter);
            plot_MSE_AMSE(dims(i),dims(i),1,dists{j},shrink,names,colors,cut);
            counter = counter+1;
        end
    end

    print -deps figure_7a
    
    r_vals = [2 4];
    h_rank_dist = figure('Position',[100,100,1300,600]);
    counter =1;
    for i=1:2
        for j=1:4
            fprintf('making plot %d/%d\n',counter,8);
            subplot(2,4,counter);
            plot_MSE_AMSE(50,50,r_vals(i),dists{j},shrink,names,colors,cut);
            counter = counter+1;
        end
    end
    
    print -deps figure_7b
    

function plot_MSE_AMSE(M,N,r,dist,shrink,names,colors,cut)
%
% Use this function to make experiments comparing MSE to AMSE, similar to Figure
% 7. 
%
assert(M<=N);
assert(r<=M);
assert(length(shrink) == length(cut));
assert(length(shrink) == length(names));

mu_vals = 1:0.05:3.6; % must have mu>1
nMonte = 50;

mse = zeros(length(mu_vals), length(shrink));

for i_mu = 1:length(mu_vals);
    mu = mu_vals(i_mu);
    dX = zeros(1,M);
    dX(1:r) = mu;
    [UX tmp1 tmp2] = svd(randn(M));
    [tmp1 tmp2 VX] = svd(randn(N));
    X = UX * [diag(dX) zeros(M,N-M)] * VX';
 



    this_mse = zeros(nMonte,length(shrink)); 
    for iMonte = 1:nMonte
        switch dist
        case 'student-t(6)'
            Z = trnd(6,M,N) / sqrt(6/4);
        case 'gaussian'
            Z = randn(M, N);
        case 'uniform'
            Z = (rand(M,N)-0.5)*sqrt(12);
        case 'bernoulli',
            Z = (rand(M,N)>0.5)*2-1;
        otherwise
            error('unknown dist')
        end
        Z = Z/sqrt(N);
        Y = X + Z;
        
        [UY DY VY] = svd(Y);
        y = diag(DY);
   
        for i_shrink =1:length(shrink)

            xhat = shrink{i_shrink}(y);
            if(cut(i_shrink))
                 xhat(r+1:end)=0;
            end
        
            DH = [ diag(xhat) zeros(M, N - M)];
            H = UY * DH * VY'; 
            this_mse(iMonte,i_shrink)  = (norm(X - H, 'fro')^2);
        end

    end
    mse(i_mu,:) = mean(this_mse);
end

beta = M/N;
I = mu_vals> beta^(0.25);
t = zeros(size(mu_vals));
w = zeros(size(mu_vals));
t(I) = sqrt((mu_vals(I) + 1./mu_vals(I)).*(mu_vals(I) + beta./mu_vals(I)));
t(~I) = 1+sqrt(beta);
w(I) = (mu_vals(I).^4 - beta) ./ ((mu_vals(I).^2) .* t(I));

for i_shrink = 1:length(shrink)
plot(mu_vals,mse(:,i_shrink), ['o' colors{i_shrink}], 'LineWidth',2,'MarkerSize',6);
hold on;
end
for i_shrink = 1:length(shrink)
mse_formula = zeros(size(mu_vals));
mse_formula = r * (shrink{i_shrink}(t).^2 - 2 * shrink{i_shrink}(t).* w + mu_vals.^2);
plot(mu_vals,mse_formula,['-' colors{i_shrink}],'LineWidth',1.5,'MarkerSize',6);
end

title(sprintf('$(%d,%d,%d)$ %s',M,N,r,dist),'Interpreter','Latex','FontSize',12);
axis tight;
xlabel('$x$','FontSize',10,'Interpreter','Latex');
ylabel('$MSE$','FontSize',10,'Interpreter','Latex');
h=legend(names,'Interpreter','Latex','Location','SouthEast');
set(h,'FontSize',15);


