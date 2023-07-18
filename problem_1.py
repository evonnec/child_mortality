## take in the csv
## for each line, find all the dimensions leading up to the max dimension size.
## calculate the ratio metric as we go thru each dimension up to the max dimension size.
## let's return the ratio metric for each dimension combination up to the max dimension size.

## later:
## write the ratio metric out, mapped also to the dimension combination

# https://stackoverflow.com/questions/64213656/how-to-get-various-combinations-of-categories-in-a-categorical-variable-and-at-t
# Digest
# Use itertools.combinations(). Further key steps:

# Apply itertools.combinations() on every possible number of dimensions to be summed up (from 1 to n_dim - 1). 
# i.e. itertools.combinations(range(1, 1+n_dim), i), for i in range(1, 1+n_dim).

# Use df.groupby(by=column_combinations).sum() to get the results from the combinations of classes automatically.

# Code
# The program consists of 3 logical parts.

# Aggregate by a class from every dimension. 
# This part basically equals to what you have done, 
# but is re-designed by a DFS method to reduce the total amount of data being processed. 

# This can be useful when there are millions of rows to be processed. 

# Later steps were also computed based on this intermediate dataset instead of the raw dataset.

# A generator to loop through dimensional combinations mentioned in Digest 1 and without explicit enumeration.

# Perform group-by computations mentioned in Digest 2 and output a list of result dataframes which can be concatenated at the end of the program.
# Caution: Be sure to test for performance and memory issues in production use.

