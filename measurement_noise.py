import math
import random
import numpy as np

def normalize_scores(scores): # Returns normalized vector with mean = 100 and SD = 15
    
    # Compute the mean and standard deviation of the input vector
    mean = np.mean(scores)
    std = np.std(scores)
    
    # Normalize the scores using the formula: normalized_score = (score - mean) * (desired_std / current_std) + desired_mean
    return [(score - mean) * (15 / std) + 100 for score in scores] 
    
def WAIS_IV_error(normalized_scores):
    # Apply noise to each score
    noisy_scores = [score + np.random.normal(0, 3+(score-np.mean(normalized_scores))/np.std(normalized_scores)) for score in normalized_scores]
    return noisy_scores
    
input = [70, 85, 100, 115, 130]
normalized_input = normalize_scores(input)
noisy_normalized_input = WAIS_IV_error(normalized_input)

print(normalized_input)
print(noisy_normalized_input)