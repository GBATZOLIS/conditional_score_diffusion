% See D. L. Donoho and M. Gavish, "The Optimal Hard Threshold for Singular
% Values is 4/sqrt(3)", http://arxiv.org/abs/1305.5870
%
% This script generate Figure 2
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


function figure2
     make_figure2([0.1 0.3 0.7 1.0]);


function make_figure2(beta_vals)
     
     assert(length(beta_vals)==4)
     
     figure2_h = figure('Position',[100,100,1900,300]);
     
     for i=1:length(beta_vals)
          this_beta = beta_vals(i);
          subplot(1,4,i)
          AMSE_plot(this_beta);
     end
 
     print -deps figure_2
    
     

function AMSE_plot(beta)
     
     assert(0<= beta)
     assert(beta<=1);
     
     r=1;
     mu_vals = 0:0.01:2.7;
     w = (8 * beta) ./ (beta + 1 + sqrt(beta.^2 + 14 * beta +1)); 
     lambdastar = sqrt(2 * (beta + 1) + w);
     
     jitter =3*[0.01,0.02,0,0.03,-0.01];
     
     names={'$\hat{X}_r$','$\hat{X}_{2.02}$','$\hat{X}_{\lambda_*}$', ...
                                 '$\hat{X}_{1+\sqrt{\beta}}$','$\hat{X}_{opt}$'};
     lines =  {'-.m','--b','-r','-.c','-k'};
     shrink{1} = @(x)( max(x, 1+ sqrt(beta)));       %trunc WRONG
     shrink{2} = @(x)(x .* (x>=2.02)); %hard
     shrink{3} = @(x)(x .* (x>=lambdastar)); %hard
     shrink{4} = @(x)(x .* (x>1+sqrt(beta))); %hard
     shrink{5} = @(x)( equalizer(x, beta)); %equalizer
     
     assert(length(shrink)==length(names));
     
     I = mu_vals> beta^(0.25);
     t = zeros(size(mu_vals));
     t(I) = sqrt((mu_vals(I) + 1./mu_vals(I)).*(mu_vals(I) + beta./mu_vals(I)));
     t(~I) = 1+sqrt(beta);
     w = (mu_vals(I).^4 - beta) ./ ((mu_vals(I).^2) .* t(I));
     
     
     hold on;
     for i=1:length(shrink)
     mse_formula = zeros(size(mu_vals));
     mse_formula(~I) = mu_vals(~I).^2 + shrink{i}(t(~I)).^2;
     mse_formula(I) = r * (shrink{i}(t(I)).^2 - 2 * shrink{i}(t(I)).* w + mu_vals(I).^2);
     plot(mu_vals,mse_formula+jitter(i),lines{i},'LineWidth',3,'MarkerSize',10);
     end
     
     grid on;
     axis tight
     
     h_legend = legend(names,'Location','NorthEast','Interpreter','Latex');
     set(h_legend,'FontSize',15);
     xlabel('$x$','FontSize',20,'Interpreter','Latex');
     ylabel('$AMSE$','FontSize',20,'Interpreter','Latex');
     ylim([0 5]);
     title(['$\beta= ' num2str(beta,2) '$'],'FontSize',20,'Interpreter','Latex');
     


function result = equalizer(x, beta)
     
     if(nargin<2)
         beta=1;
     end
     
     assert( (beta > 0) & (beta <= 1));
     assert(any(x<0) == 0);
     
     result = sqrt( (x.^2 - (1+beta)).^2 - 4*beta ) ./ x;
     result( x < (1+sqrt(beta)) ) = 0;
     

