% See D. L. Donoho and M. Gavish, "The Optimal Hard Threshold for Singular
% Values is 4/sqrt(3)", http://arxiv.org/abs/1305.5870
%
% This script generate Figure 3
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


function figure3

    mu_vals = 0:0.001:3;
    
    lam_soft = 2
    lam_hard = 4/sqrt(3)
    mu = @(x)((x+sqrt(x^2-4))/2)
    
    figure3_h = figure('Position',[100,100,1200,1000]);
    
    soft_mse = 6+3./(mu_vals.^2)-8./mu_vals;
    I_soft = mu_vals < mu(lam_soft);
    soft_mse(I_soft) = mu_vals(I_soft).^2;
    [m i_soft] = max(soft_mse./mu_vals);
    
    
    hard_mse = 2+3./(mu_vals.^2);
    I_hard = mu_vals<mu(lam_hard);
    hard_mse(I_hard) = mu_vals(I_hard).^2;
    [m i_hard] = max(hard_mse./mu_vals);
    
    optimal_mse = 2 - 1./mu_vals.^(2);
    I_op = mu_vals <1;
    optimal_mse(I_op) = mu_vals(I_op).^2;
    [m i_optimal] = max(optimal_mse./mu_vals);
    
    plot(mu_vals,hard_mse,'-r',mu_vals,soft_mse,'-b',mu_vals,optimal_mse,'g','LineWidth',4)
    hold on
    
    plot(mu_vals(1:i_hard),(hard_mse(i_hard)/mu_vals(i_hard))*mu_vals(1:i_hard),'--r',...
         mu_vals(1:i_soft),(soft_mse(i_soft)/mu_vals(i_soft))*mu_vals(1:i_soft),'--b',...
         mu_vals(1:i_optimal), ...
         (optimal_mse(i_optimal)/mu_vals(i_optimal))*mu_vals(1:i_optimal),'--g','LineWidth',2);
    
         xlabel('$x$','FontSize',30,'Interpreter','Latex');
         ylabel('$AMSE$','FontSize',30,'Interpreter','Latex');
         set(gca,'FontSize',15)
    
    h_legend=legend({'$\hat{X}_{\lambda_*}$','$\hat{X}_{s_*}$', ...
         '$\hat{X}_{optimal}$'},'Location','NorthWest','Interpreter','Latex');
    set(h_legend,'FontSize',30);
   
    print -deps figure_3
