%% Given s1, s2, s3, ..., s_L (L variables in total), 
% evaluate dual total correlation (Eq. 5 in paper) sum(H(s\s_i)) - (L-1)*H(s1,s2,s3,..,s_L) 
% In this example, we consider 5 independent information sources, each follows a Gaussian
% distribution, the ground truth total correlation is zero.

%% if you use the code, please cite the following manuscript:
% Yu, Shujian, et al. "Measuring dependence with matrix-based entropy functional." 
% Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 12. 2021.

%%
clear
clc

% we consider the total correlation evaluated by our estimator and by the basic 
% Shannon's discrete entropy functional w.r.t. different number of samples 
nsamples = [100 200 300 400 500 600 700 800 900 1000];
NDTC_Renyi = zeros(1,10);
DTC_Renyi = zeros(1,10);

for i=1:10
%% generate data
dim = 5;
type = '1';
data = synthetic_data_generation(nsamples(i),dim,type);

%% TC estimation using matrix-based Renyi definition
addpath(genpath('Renyi entropy'))

%% kernel width estimation ('Silverman','kNN','graph')
sigma = kernel_width_estimation(data(:,1),'Silverman');
sigma = 10*sigma;

%%
% sigma = 1;
alpha = 1.01;

[ndtc_Renyi,dtc_Renyi] = normalize_dual_total_correlation_estimation(data,sigma,alpha);
NDTC_Renyi(i) = ndtc_Renyi;
fprintf('Normalized dual TC estimated with matrix-based Renyi''s definition with %3d samples is %6.4f \n',nsamples(i),ndtc_Renyi);

DTC_Renyi(i) = dtc_Renyi;
fprintf('dual TC estimated with matrix-based Renyi''s definition with %3d samples is %6.4f \n',nsamples(i),dtc_Renyi);

end

%% plot results
plot(nsamples,DTC_Renyi,'-kp','MarkerSize',12,'LineWidth',2);hold on
plot(nsamples,NDTC_Renyi,'--bs','MarkerSize',12,'LineWidth',2);
plot(nsamples,zeros(1,10),'r','MarkerSize',12,'LineWidth',2);
ylim([0 1]);
legend('Dual Total Correlation (Renyi)','Dual Normalized Total Correlation (Renyi)','Ground truth');
set(gca, 'FontSize', 12);
set(gca, 'FontName', 'Arial');
xlabel('Number of samples in each variable');
ylabel('Total correlation');