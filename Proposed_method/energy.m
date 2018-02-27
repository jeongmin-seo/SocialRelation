% function [sum_erfc_energy,sum_log_energy,log_sum_energy, erfc_sum_energy,log_dis] = energy(Scale_img,...
%     Scale_ratio, dist, Exp1, Exp2, Exp3, Exp4, Exp5, Exp6, Exp7, Exp8)

function [sum_erfc_energy,sum_log_energy,log_sum_energy, erfc_sum_energy,log_dis] = energy(alpha,beta, Scale_thre, Scale_img,...
    Scale_ratio, dist, Exp1, Exp2, Exp3, Exp4, Exp5, Exp6, Exp7, Exp8)

%alph1 -distance, beta - relation

function_thre = 2;
% image 1 = 5.7
% image 2 = 6.9

%% distance term
dis = dist/Scale_img;
log_dis = log(dis);
% erfc_dis = erfc(dis);

%% test1
test_EXP1 = Exp1;
test_EXP2 = Exp2;
test_EXP3 = Exp3;
test_EXP4 = Exp4;
test_EXP5 = Exp5;
test_EXP6 = Exp6;
test_EXP7 = Exp7;
test_EXP8 = Exp8;

test_sum = test_EXP1+test_EXP2+test_EXP3+test_EXP4+test_EXP5+test_EXP6+test_EXP7+test_EXP8;

log_sum_EXP = log(test_sum);
erfc_sum_EXP = erfc(test_sum);


%% test 2
test_EXP1 = log(Exp1);
test_EXP2 = log(Exp2);
test_EXP3 = log(Exp3);
test_EXP4 = log(Exp4);
test_EXP5 = log(Exp5);
test_EXP6 = log(Exp6);
test_EXP7 = log(Exp7);
test_EXP8 = log(Exp8);

sum_log_EXP = test_EXP1+test_EXP2+test_EXP3+test_EXP4+test_EXP5+test_EXP6+test_EXP7+test_EXP8;
% sum_log_EXP = -sum_log_EXP/2;
% sum_log_EXP = sum_log_EXP/2;

sum_log_EXP = -sum_log_EXP;

%% test 3
test_EXP1 = erfc(Exp1);
test_EXP2 = erfc(Exp2);
test_EXP3 = erfc(Exp3);
test_EXP4 = erfc(Exp4);
test_EXP5 = erfc(Exp5);
test_EXP6 = erfc(Exp6);
test_EXP7 = erfc(Exp7);
test_EXP8 = erfc(Exp8);

sum_erfc_EXP = test_EXP1+test_EXP2+test_EXP3+test_EXP4+test_EXP5+test_EXP6+test_EXP7+test_EXP8
sum_erfc_EXP = sum_erfc_EXP;


%% final

% energy = alpha1*(1/ratio_h)*(1/ratio_w)*(1/dist)+...
%          alpha2*(Exp1)+alpha3*(Exp2)+alpha4*(Exp3)+...
%          alpha5*(Exp4)+alpha6*(Exp5)+alpha7*(Exp6)+...
%          alpha8*(Exp7)+alpha9*(Exp8)+alpha10;
 
% energy = alpha1*dis2 +alpha2*sum_log_EXP+thre

% sum_erfc_energy = alpha*log_dis + beta*sum_erfc_EXP + function_thre ;
% sum_log_energy = alpha*log_dis + beta*sum_log_EXP + function_thre ;
% log_sum_energy = alpha*log_dis + beta*log_sum_EXP + function_thre ;
% erfc_sum_energy = alpha*log_dis + beta*erfc_sum_EXP + function_thre; 

sum_erfc_energy = beta*log_dis + (1-beta)*sum_erfc_EXP + function_thre ;
sum_log_energy = beta*log_dis +(1-beta)*sum_log_EXP + function_thre ;
log_sum_energy = beta*log_dis + (1-beta)*log_sum_EXP + function_thre ;
erfc_sum_energy = beta*log_dis + (1-beta)*erfc_sum_EXP + function_thre; 

%% Scale_ratio

% Scale_ratio = -log(Scale_ratio);
if Scale_ratio > Scale_thre
    sum_erfc_energy = 10 ;
    sum_log_energy = 5 ;
    log_sum_energy = 10 ;
    erfc_sum_energy = 10 ;
end
% 
% sum_erfc_energy = alpha1*log_dis + alpha2*sum_erfc_EXP + Scale_ratio + function_thre ;
% sum_log_energy = alpha1*log_dis + alpha2*sum_log_EXP +Scale_ratio +  function_thre ;
% log_sum_energy = alpha1*log_dis + alpha2*log_sum_EXP + Scale_ratio +  function_thre ;
% erfc_sum_energy = alpha1*log_dis + alpha2*erfc_sum_EXP + Scale_ratio + function_thre; 
% % 
% 
% sum_erfc_energy =-( alpha1*log_dis + alpha2*sum_erfc_EXP + Scale_ratio + function_thre );
% sum_log_energy = -(alpha1*log_dis + alpha2*sum_log_EXP +Scale_ratio +  function_thre );
% log_sum_energy = -(alpha1*log_dis + alpha2*log_sum_EXP + Scale_ratio +  function_thre );
% erfc_sum_energy = -(alpha1*log_dis + alpha2*erfc_sum_EXP + Scale_ratio + function_thre); 
