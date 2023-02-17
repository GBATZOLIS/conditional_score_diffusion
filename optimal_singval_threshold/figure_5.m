% See D. L. Donoho and M. Gavish, "The Optimal Hard Threshold for Singular
% Values is 4/sqrt(3)", http://arxiv.org/abs/1305.5870
%
% This script generate Figure 5
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


function figure5

    beta_vals = 0.01:0.01:1;
    npoints = 10000;
    lam1_vals = 2.02 * ones(size(beta_vals));
    
    lam2_vals = sqrt(2*(beta_vals+1) + (8*beta_vals) ./ ...
                                    ((beta_vals+1)+sqrt(beta_vals.^2+14*beta_vals+1)));
    
    lf1 = zeros(size(beta_vals));
    lf2 = zeros(size(beta_vals));
                                    
    for i=1:length(beta_vals)
        c = beta_vals(i);
        mu_vals = linspace(c^(1/4),5,npoints);
        shrink1 = @(x)(x .* (x>=lam1_vals(i)));
    
        t = sqrt((mu_vals + 1./mu_vals).*(mu_vals + c./mu_vals));
        w = (mu_vals.^4 - c) ./ ((mu_vals.^2) .* t);
    
        lf1(i) = max((shrink1(t).^2 - 2 * shrink1(t).* w + mu_vals.^2));
        shrink2 = @(x)(x .* (x>=lam2_vals(i)));
        lf2(i) =  max((shrink2(t).^2 - 2 * shrink2(t).* w + mu_vals.^2));
    end
    
    
    figure5_h = figure('Position',[100,100,1000,600]);
    
    plot(beta_vals,lf1,'-r',beta_vals,lf2,'-b','LineWidth',5);
    xlabel('$\beta$','FontSize',20,'Interpreter','Latex');
    ylabel('AMSE','FontSize',20,'Interpreter','Latex');
    grid on;
    h_legend=legend({'$\max_{\,x}\, \mathbf{M}(\hat{X}_{2.02}\,,\,x)$', ...
              '$\max_{\,x}\, \mathbf{M}(\hat{X}_{\lambda_*(\beta)}\,,\,x)$'}, ...
              'Interpreter','Latex','Location','NorthWest');
    set(h_legend,'FontSize',20);
    
    print -deps figure_5
