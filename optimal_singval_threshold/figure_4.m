% See D. L. Donoho and M. Gavish, "The Optimal Hard Threshold for Singular
% Values is 4/sqrt(3)", http://arxiv.org/abs/1305.5870
%
% This script generate Figure 4
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


function figure4
    
    beta = 0:0.01:1;
    
    w = (8 * beta) ./ (beta + 1 + sqrt(beta.^2 + 14 * beta +1)) 
    
    lambda_opt = sqrt(2 * (beta + 1) + w);
    
    figure4_h = figure('Position',[100,100,800,600]);

    plot(beta,lambda_opt,'-b','LineWidth',4);
    xlabel('$\beta$','FontSize',20,'Interpreter','Latex');
    ylabel('$\lambda$','FontSize',20,'Interpreter','Latex');
    
    hold on; 
    grid on;
    plot(beta,1+sqrt(beta),'-r','LineWidth',5);
    plot(beta,2.02*ones(size(beta)),'--g','LineWidth',5);
    h_legend=legend({'$\lambda_*(\beta)$','$1+\sqrt{\beta}$','2.02'},'Location','NorthWest','Interpreter','Latex');
    set(h_legend,'FontSize',15);
   
    print -deps figure_4
