function data = synthetic_data_generation(num_samples,dim,type)

data = zeros(num_samples,dim);

if strcmp(type,'1') == 1 % all variables are independent
    for j=1:dim
        data(:,j) = randn(num_samples,1);
    end
elseif strcmp(type,'2') == 1 % all variables linearly depents on variable 1 
    data(:,1) = randn(num_samples,1);
    for j=2:dim
        data(:,j) = data(:,1);
    end
elseif strcmp(type,'3') == 1 % all variables nonlinearly depents on variable 1 
    data(:,1) = randn(num_samples,1);
    for j=2:dim
        data(:,j) = data(:,1).^2 + data(:,1);
    end
elseif strcmp(type,'4') == 1 % last variable depends on all other variables
    data(:,1:end-1) = randn(num_samples,dim-1);
    data(:,end) = mean(data(:,1:end-1).^2,2);
end

end