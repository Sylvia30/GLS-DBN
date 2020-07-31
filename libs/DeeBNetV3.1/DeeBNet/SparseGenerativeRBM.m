%****************************In the Name of God****************************
% Different methods are proposed to build sparse RBMs. In all proposed
% methods, learning algorithm in RBM has been changed to enforce RBM to
% learn sparse representation. The goal of sparsity in RBM is that the most
% of hidden units has zero values and this is equivalent to tend activation
% probability of hidden units to zero. Sparse generative RBM is an
% sparse RBM for generative purposes like feature extraction, etc.

% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our web page.
%
% The programs and documents are distributed without any warranty, express
% or implied.  As the programs were written for research purposes only,
% they have not been tested to the degree that would be advisable in any
% important application.  All use of these programs is entirely at the
% user's own risk.

% CONTRIBUTORS
%	Created by:
%   	Mohammad Ali Keyvanrad (http://ceit.aut.ac.ir/~keyvanrad)
%   	01/2016
%           LIMP(Laboratory for Intelligent Multimedia Processing),
%           AUT(Amirkabir University of Technology), Tehran, Iran
%**************************************************************************
%SparseGenerativeRBM Class
classdef SparseGenerativeRBM<GenerativeRBM & SparseRBM
    
    %% PUBLIC METHODS ------------------------------------------------------
    methods (Access=public)
        %Constructor
        function obj=SparseGenerativeRBM(rbmParams)
            obj=obj@GenerativeRBM(rbmParams);
            obj=obj@SparseRBM(rbmParams);
        end %End of Constructor function
        
    end %End PUBLIC METHODS
    
    %% PROTECTED METHODS --------------------------------------------------
    methods (Access=protected)
        
        function [deltaWeightReg,deltaVisBiasReg,deltaHidBiasReg]=getRegularizationGradient(obj,batchData,posHid)
            switch obj.rbmParams.sparsityMethod
                % based on paper [1]
                case 'quadratic'
                    term1=obj.rbmParams.sparsityTarget-posHid;
                    term2=1/size(batchData,1)*(batchData'*((posHid).*(1-posHid)));
                    term3=mean((posHid).*(1-posHid));
                    deltaHidBiasReg=obj.rbmParams.sparsityCost.*mean(term1).*term3;
                    deltaWeightReg=obj.rbmParams.sparsityCost*repmat(mean(term1),size(batchData,2),1).*term2;
                % based on paper [2]
                case 'rateDistortion'
                    term2=-1/size(batchData,1)*(batchData'*((posHid).*(1-posHid)));
                    term3=-mean((posHid).*(1-posHid));
                    deltaHidBiasReg=obj.rbmParams.sparsityCost*term3;
                    deltaWeightReg=obj.rbmParams.sparsityCost*term2;
                 % based on our paper
                case 'normal'
                    term1=obj.rbmParams.sparsityTarget-posHid;
                    term2=1/size(batchData,1)*(batchData'*((posHid).*(1-posHid)));
                    term3=mean((posHid).*(1-posHid));
                    term4=normpdf(mean(posHid),obj.rbmParams.sparsityTarget,sqrt(obj.rbmParams.sparsityVariance));
                    deltaHidBiasReg=obj.rbmParams.sparsityCost*mean(term1).*term3.*term4;
                    deltaWeightReg=obj.rbmParams.sparsityCost*repmat(mean(term1),size(batchData,2),1).*term2.*repmat(term4,size(batchData,2),1);
                 case 'Laplace'
                    s = size(batchData);
                    po = size(posHid);
                    term1 = mean(posHid); 
                    term2 = mean((posHid).*(1-posHid)); 
                    p = 0.06;
                    
                    for i = 1:po(2)
                        if term1(1,i) < p
                            term3(1,i) = 0.5*exp(-(p-mean(posHid(:,i))));
                        else
                            term3(1,i) = -0.5*exp(-(mean(posHid(:,i))-p));
                        end  
                    end
                    
                    deltaHidBiasReg = obj.rbmParams.sparsityCost*term3 .* term2;
                    deltaWeightReg = obj.rbmParams.sparsityCost *(1/size(batchData,1))*(batchData'*((posHid).*(1-posHid))) .* repmat(term3,s(2),1); 
                
                case 'cauchy'
                    a = 1;
                    b = 1;
                    term1 = mean((posHid).*(1-posHid)); 
                    term2 = term1-0.1;
                    term3 = term2./(((term2.^2+b^2)).^2);
                    
                    
                    
                    deltaWeightReg = obj.rbmParams.sparsityCost *(1/size(batchData,1))*(batchData'*((posHid).*(1-posHid))) .* repmat(term3,size(batchData,2),1); 
                    
                    deltaHidBiasReg = obj.rbmParams.sparsityCost*term3 .* term1;
                    
                case 'lasso'
                    %分组
                    n = 10;
                    %重叠比率
                    rate = 0.2;
                    %normP = sum(sqrt(sum(p.^2)))
                    %计算penalty
                    %每组的神经元数目
                    num_overlap = ((1+rate)*size(posHid,2))/n;
                    %重叠的节点数目
                    num_p = num_overlap* rate;
                    index = 1:size(posHid,2);
                    instance = num_overlap-num_p-1;
                        start = 1; 
                    for i = 1:n
                        group(i,:) = index(start:start+num_overlap-1);
                        start = start + instance;
                    end
                    %重叠索引
                    for i = 1:n
                        temp = group(i,:);
                        if i == 1
                            index_ogroup(i) = {temp(instance+1:instance+num_p)};
                            index_nextogroup(i) = {temp(instance+1:instance+num_p)};
                        elseif i == n
                            index_ogroup(i) = {temp(1:num_p)};
                            index_nextogroup(i) = {[]};
                        else
                             index_ogroup(i) = {[temp(1:num_p),temp(instance+1:instance+num_p)]};
                             index_nextogroup(i) = {temp(instance+1:instance+num_p)};
                        end

                        index_nogroup(i) = {setdiff(temp,cell2mat(index_ogroup(i)))};  
                    end
                    %计算penalty
                    term1 = mean((posHid).*(1-posHid)); 

                    for i = 1:n
                        pen_nogroup(i) = sqrt(sum(term1(cell2mat(index_nogroup(i))).^2));
                        pen_ogroup(i) =  sqrt(sum(term1(cell2mat(index_ogroup(i))).^2)) + sqrt(sum(term1(cell2mat(index_nextogroup(i))).^2));
                    end

                    term2 = (posHid.^2) .* (1-posHid);
                    total_op = [];
                    for i = 1:n-1
                        temp = intersect(cell2mat(index_ogroup(i)),cell2mat(index_ogroup(i+1)));
                        total_op = [total_op,temp];
                    end

                    total_nop = setdiff(index,total_op); 

                    P = zeros(1,size(posHid,2));
                    P(total_op) = sum(pen_ogroup);
                    P(total_nop) = sum(pen_nogroup);

                    PP=repmat(P,size(batchData,1),1);%%%%%%%%%%%%%%%%%%%
                    term3 = 0.01* term2 ./ PP;

                    deltaHidBiasReg = obj.rbmParams.sparsityCost*mean(term3);
                    deltaWeightReg = obj.rbmParams.sparsityCost *(1/size(batchData,1))*(batchData'* term3); 
                    
%                     s_term2=-1/size(batchData,1)*(batchData'*((posHid).*(1-posHid)));
%                     s_term3=-mean((posHid).*(1-posHid));
%                     s_deltaHidBiasReg=obj.rbmParams.sparsityCost*s_term3;
%                     s_deltaWeightReg=obj.rbmParams.sparsityCost*s_term2;
%                     
%                     deltaHidBiasReg = deltaHidBiasReg+ s_deltaHidBiasReg;
%                 
%                     deltaWeightReg = deltaWeightReg + s_deltaWeightReg;


                    term1 = []; term2 = []; term3 = [];
                    a = 1;
                    b = 0.8;
                    term1 = mean((posHid).*(1-posHid)); 
                    term2 = term1-0.2;
                    term3 = term2./(((term2.^2+b^2)).^2);
                    
                    
                    
                    deltaWeightReg = obj.rbmParams.sparsityCost *(1/size(batchData,1))*(batchData'*((posHid).*(1-posHid))) .* repmat(term3,size(batchData,2),1) + deltaWeightReg; 
                    
                    deltaHidBiasReg = obj.rbmParams.sparsityCost*term3 .* term1 + deltaHidBiasReg;
                
                otherwise
                    error('Your sparsity method is not defined');
            end            
            deltaVisBiasReg=0;
        end %End of getRegularizationGradient function
        
    end %End PROTECTED METHODS
   
end %End SparseGenerativeRBM class

