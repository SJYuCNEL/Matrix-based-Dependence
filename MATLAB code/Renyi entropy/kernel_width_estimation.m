function sigma = kernel_width_estimation(data,opt)

if strcmp(opt,'graph') == 1
    % estimate sigma with 10 to 20 percent of the total (median) range of the Euclidean
    % distances between all pairwise data points (pls refer details in the paper)
    dis_range = pdist(data(:,1)); 
    dis_range = sort(dis_range); 
    sigma = dis_range(round(0.05*numel(dis_range))); 
elseif strcmp(opt,'Silverman') == 1
    % estimate sigma with silverman's rule of thumb
    sigma = std(data(:,1))*(4/3/numel(data(:,1)))^(1/5);
    sigma = sigma/sqrt(2);
elseif strcmp(opt,'kNN') == 1
    % estimate sigma with kNN search
    Mdl = KDTreeSearcher(data);
    k = 5;
    IdxNN = knnsearch(Mdl,data,'K',k+1); 
    IdxNN = IdxNN(:,2:end);
    IdxNN = IdxNN';
    IdxNN = IdxNN(:);
    Idx_ref = repmat(1:size(data,1),[k 1]);
    Idx_ref = Idx_ref(:);
    
    diff_data = data(IdxNN,:) - data(Idx_ref,:);
    sigma = mean(mean(diff_data.^2,2));

end

end