clc;clear;close all;

%%
%相关性分析


%UPDRS=[10;16;32;20;17;13;8;9;12;28;22;41;27;30;15;15;14;27;3;23;26;13;32;14;35;16;21;22;19;26;20;8;7;40;20;15;19;41;33;25;47;46;10;16;8;13;6;43];
%PDRP_Score=[1.55;0.660;1.70;0.860;1.60;1.46;1.43;0.530;0.890;1.96;1.28;1.22;0.350;0.990;1.52;1.61;2.22;1.79;0.770;0.940;1.99;1.78;2.29;1.45;0.170;1.75;3.02;1.51;1.43;2.11;1.21;0.860;2.02;2.99;1.54;2.37;1.60;2.08;1.03;2.09;2.63;1.62;1.27;1.67;1.04;1.08;1.20;2.28];

load DiseaseDuration.mat;
load HY.mat;
load UPDR.mat;
load pdfeature.mat

load DRdata.mat;
load bpfeature1.mat;
PDdata = DRdata(:,1:103)';
pddata = pddata';
for i = 1:500
    [r,p]=corr(UPDR',c(:,i));
    %figure
    figure('Visible','off');
    plot(UPDR',c(:,i),'o');
    lsline
    title({'Correlation between DiseaseDuration and LLE feature',['R=',num2str(roundn(r,-2)),'  P=',num2str(roundn(p,-3))]},'FontSize',14,'FontWeight','bold')
    xlabel('DiseaseDuration','FontWeight','bold');
    ylabel('LLE feature','FontWeight','bold');
    name = strcat('C:\Users\ST\Desktop\smartsleep\corr\',num2str(i),'.jpg');
    saveas(gcf,name); 
end



% title({'Correlation between PDRP Score and UPDRS',['R=',num2str(roundn(r,-2)),'  P=',num2str(roundn(p,-3))]},'FontSize',14,'FontWeight','bold')
% xlabel('UPDRS','FontWeight','bold');
% ylabel('PDRP Score','FontWeight','bold');
