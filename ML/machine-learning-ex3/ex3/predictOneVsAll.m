function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
%tmp_p= zeros(size(X, 1), 1);
% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%
%       for the value is only 0,1 after round or >=0.5 so we can get the index 
%       by max?     

prob = sigmoid(all_theta * X');% 10x5000
[val,p] = max(prob,[],1);  
%get the max possibility data. But it is not open(means  here we used 
%the information that the predict data is only 1~10. however in the 
%real application we can not limit it, so it is only about 89%.)
%If in the open conditions, we should use following way as I designed.
p = p';

%for c=1:num_labels
  
%  tmp_p=(sigmoid(X * (all_theta(c,:))')>=0.5);

%  for i=1:m
%    if (p(i)==0) && (tmp_p(i)==1)
%      p(i)=c;
%    end
%  end
%  Val=1;
%end

%If we use the following way to do the work it will overwrite some vector that 
%has bee recogonized as one figure to others. e.g. 1st round we get the 497 1 
%but after total round it will become 487. So we change the way. 
  %for i=1:m
  %  if tmp_p(i)==1
  %    p(i)=c;
  %  end
  %end
%===============================
% =========================================================================

end
