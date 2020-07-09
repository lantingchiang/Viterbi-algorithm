% Input: A - (n*n) state transition probability matrix
%        B - (n*m) state observation probability matrix
%        p - (1*n) starting state probability
%        w - (1*T) indices associated with observed sequence of states
% Output: I - (1*T) indices associated with sequence of hidden states that
%                   give highest probability of producing observed states
%         maxscore - highest probability score found at time T
function [I, maxscore] = modified_viterbi(A, B, p, w)
    n = length(A);
    T = length(w);
    score = zeros(n, T);
    pred = zeros(n, T);
    I = zeros(1, T);
    
    % find starting scores (for output sequence of length 1)
    for j = 1 : n
        score(j, 1) = log(p(j)) + log(B(j, w(1)));
    end
    
    % forward phase to find best starting scores
    for t = 2 : T
        for j = 1 : n
            tscore = zeros(1, n);
            for k = 1 : n
                tscore(k) = score(k, t-1) + log(A(k, j)) + log(B(j, w(t)));
            end
            [maxvalue, index] = max(tscore);
            score(j, t) = maxvalue;
            pred(j, t) = index;
        end
    end
    
    % second phase to retrive the optimal path
    [maxvalue, index] = max(score(:, T));
    I(T) = index;
    maxscore = maxvalue;
    for t = T : -1 : 2
        I(t-1) = pred(I(t), t);
    end            
end