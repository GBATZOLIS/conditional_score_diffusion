% See D. L. Donoho and M. Gavish, "The Optimal Hard Threshold for Singular
% Values is 4/sqrt(3)", http://arxiv.org/abs/1305.5870
%
% This script generate Figure 6
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


function figure6
    
    figure6_h = figure('Position',[100,100,1600,250]);
    
    lambda_vals = [2.02 2.15 4/sqrt(3) 2.45 2.75];
    mu_vals = 1:0.001:2.5;
    linew =5;
    slinew = 1.5;
    for i=1:length(lambda_vals);
        lambda = lambda_vals(i);
        cutoff = (lambda + sqrt(lambda^2-4))/2;
        I = mu_vals + 1./mu_vals > lambda;
        subplot(1,5,i);
        plot(mu_vals, mu_vals.^2, '--g','LineWidth',slinew);
        hold on;
        plot(mu_vals, 2+3./(mu_vals.^2),'--b','LineWidth',slinew);
        plot(mu_vals(~I),mu_vals(~I).^2,'-g','LineWidth',linew);
        plot(mu_vals(I), 2+3./(mu_vals(I).^2),'-b','LineWidth',linew);
        line([cutoff cutoff],[1 6],'LineWidth',slinew,'Color','r');
        grid on;
        axis tight;
        xlabel('$x$','FontSize',15,'Interpreter','Latex');
        ylabel('$AMSE$','FontSize',15,'Interpreter','Latex');
        title(['$\lambda= ' num2str(lambda,3) '$'],'FontSize',15,'Interpreter','Latex'); 
    end
    
    print -deps figure_6
