datamode = 'difficile';

model = load(strcat('hinge_model_', datamode, '.mat'));
M = model.M;

pairs = data_test.pairs;
pairs = pairs + 1;

[n, d] = size(data_test.X);
m = size(pairs, 1);
dist = zeros(m, 1);

X = data_test.X;

for i = 1:m
    x1 = X(pairs(i, 1), :);
    x2 = X(pairs(i, 2), :);
    dist(i) = 1.0 - ((x1 * M * x2') / (sqrt(x1 * M * x1') .* sqrt(x2 * M * x2')));
end 

fileID = fopen(strcat('DINH_', datamode, '.txt'), 'w');
fprintf(fileID, '%f\n', dist);
fclose(fileID);