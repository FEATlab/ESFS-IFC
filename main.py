import os
import glob
import scipy.io as sio
import numpy as np
import pandas as pd
from testing import testing

# Specify folder paths
folder = 'processed_datasets'
new_folder = r'results\MatSummary'
new2_folder = r'results\CsvSummary'
Metric = 'NMI'  # Evaluation Metric

# Get a list of all .mat files in the folder
files = glob.glob(os.path.join(folder, '*.mat'))

# Create target folders if they do not exist
os.makedirs(new_folder, exist_ok=True)
os.makedirs(new2_folder, exist_ok=True)

# Define the path for the summary CSV file
csv_file_path = os.path.join(new2_folder, 'results_summary.csv')

# Initialize a list to store results for each dataset
results = []

for file in files:
    # Get the name of the dataset
    dataset_name = os.path.splitext(os.path.basename(file))[0]

    # Load data
    data = sio.loadmat(file)
    train_data = data.get('train_data')
    test_data = data.get('test_data')

    # Initialize parameters for metrics collection
    svmSOL = []
    knnSOL = []
    J_SOL = []
    TT = []
    AUCSOL = []
    A = []
    B = []
    G = []
    first_iter_acc_list = []  # Store training accuracy of the first iteration for each run
    first_iter_acc_list_test = []  # Store testing accuracy of the first iteration
    first_iter_features_list = []  # Store number of features selected in the first iteration
    train_acc_all = []  # Store training accuracy curves for each run
    test_acc_all = []  # Store testing accuracy curves for each run
    final_avg_train_acc = 0
    final_avg_test_acc = 0
    ga_call_count_list = []
    select_10_features = []

    RUNTIME = 30

    # Loop for the specified number of runs
    for _ in range(RUNTIME):
        print(f"Currently processing dataset: {dataset_name}")
        print(r"Run # {}".format(_ + 1))

        # execute the feature selection and evaluation pipeline
        SVMAccuracy, KNNAccuracy, AUC, d_feature, tt1, gbest, ggbest, AC, first_iter_acc, m_acc, first_iter_acc_test, svm_accuracies, Firstfeatures, ga_call_count, selected_10_features = testing(
            train_data, test_data,
            dataset_name)

        # Collect results from the current run
        svmSOL.append(SVMAccuracy)
        knnSOL.append(KNNAccuracy)
        AUCSOL.append(AUC)
        J_SOL.append(d_feature)
        TT.append(tt1)
        A.append(gbest)
        B.append(ggbest)
        G.append(AC)
        first_iter_acc_list.append(first_iter_acc)
        first_iter_acc_list_test.append(first_iter_acc_test)
        first_iter_features_list.append(Firstfeatures)
        train_acc_all.append(m_acc)
        test_acc_all.append(svm_accuracies)
        ga_call_count_list.append(ga_call_count)
        select_10_features.append(selected_10_features)

    # Convert lists to arrays and calculate mean values across iterations
    avg_10_features = np.mean(np.array(select_10_features), axis=0)
    avg_train_acc = np.mean(np.array(train_acc_all), axis=0)
    avg_test_acc = np.mean(np.array(test_acc_all), axis=0)
    final_avg_train_acc = np.mean(avg_train_acc)
    final_avg_test_acc = np.mean(avg_test_acc)
    ga_call_count_mean = np.mean(ga_call_count_list)

    # Calculate the average accuracy for each of the runs
    test_acc_30_runs_avg = [np.mean(single_run_acc) for single_run_acc in test_acc_all]

    # Calculate final means and standard deviations for reporting
    av_Acsvm = np.mean(svmSOL)
    av_Acknn = np.mean(knnSOL)
    av_AUC = np.mean(AUCSOL)
    fc_Acsvm = np.std(svmSOL)
    fc_Acknn = np.std(knnSOL)
    av_d = np.mean(J_SOL)
    aTT = np.mean(TT)
    total_time = aTT * RUNTIME

    # Save processed data to a .mat file in the summary folder
    new_filename = os.path.join(new_folder, os.path.basename(file))
    sio.savemat(new_filename, {
        'svmSOL': svmSOL,
        'knnSOL': knnSOL,
        'AUCSOL': AUCSOL,
        'J_SOL': J_SOL,
        'TT': TT,
        'A': A,
        'B': B,
        'G': G,
        'av_Acsvm': av_Acsvm,
        'av_Acknn': av_Acknn,
        'av_AUC': av_AUC,
        'fc_Acsvm': fc_Acsvm,
        'fc_Acknn': fc_Acknn,
        'av_d': av_d,
        'aTT': aTT
    })

    # Convert results into comma-separated strings for CSV storage
    first_iter_str = ','.join(map(str, first_iter_acc_list))
    first_iter_str_test = ','.join(map(str, first_iter_acc_list_test))
    first_iter_features_list_str = ','.join(map(str, first_iter_features_list))
    avg_train_acc_str = ','.join(map(str, avg_train_acc))
    avg_test_acc_str = ','.join(map(str, avg_test_acc))
    avg_10_features_str = ','.join(map(str, avg_10_features))

    # Detailed results for each of the runs
    svm_results_str = ','.join(map(str, test_acc_30_runs_avg))
    features_results_str = ','.join(map(str, J_SOL))
    time_results_str = ','.join(map(str, TT))

    # Append aggregated results for this dataset to the final list
    results.append([
        dataset_name,
        final_avg_train_acc,
        final_avg_test_acc,
        av_Acsvm,
        av_d,
        aTT,
        total_time,
        ga_call_count_mean,
        first_iter_str,
        first_iter_str_test,
        first_iter_features_list_str,
        avg_train_acc_str,
        avg_test_acc_str,
        avg_10_features_str,
        svm_results_str,
        features_results_str,
        time_results_str
    ])

    print(f"Completed dataset: {dataset_name}\n{'=' * 50}")

# Define headers for the summary CSV file
header = [
    Metric,
    'Training_Accuracy',
    'Testing_Accuracy',
    'Avg_SVM_Accuracy',
    'Avg_Selected_Features',
    'Avg_Run_Time',
    'Total_Run_Time',
    'GA_Call_Count',
    'FirstIterAcc',
    'FirstIterAccTest',
    'First_iter_features',
    'AvgTrainAcc',
    'AvgTestAcc',
    'Avg10Features',
    'all_Runs_SVM_Acc',
    'all_Runs_Feature_Count',
    'all_Runs_Run_Time'
]

# Save all results to a CSV file
df = pd.DataFrame(results, columns=header)
df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
print(f"All datasets processed. Results saved to {csv_file_path}")