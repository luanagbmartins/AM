python -m luigi --module train NonparametricRegression --selected-model running_mean --n-jobs 7 --local-scheduler
python -m luigi --module train NonparametricRegression --selected-model gaussian_kernel --n-jobs 7 --local-scheduler