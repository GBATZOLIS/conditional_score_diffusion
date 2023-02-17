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


function figure1
    
    rng default; rng(42);
    
    y=svd(diag([[1.7 2.5] zeros(1,98)])+randn(100)/sqrt(100));
    
    figure1_h = figure('Position',[100,100,1000,400]);
  
    subplot(1,2,1); plot(1:length(y),y,'o','LineWidth',1.5);
    set(gca,'FontSize',15)
    subplot(1,2,2); hist(y,15);
    set(gca,'FontSize',15)

    
    print -deps figure_1
