import torch
import pandas as pd
# "Avg_NormalNN",
Approaches = ["NormalNN", "EWC",
              "SI", "L2", "Naive_Rehearsal_1100", "Naive_Rehearsal_4400",
              "MAS", "GEM_1100", "GEM_4400"
              ]
REPEAT = 10
OutDirPath = "/home/hikmat/Desktop/JWorkspace/CL/Continuum/ContinuumBenchmarks/MNIST/Continual-Learning-Benchmark/scripts/outputs/permuted_MNIST_incremental_domain_10"
# OutDirPath = "/home/hikmat/Desktop/JWorkspace/CL/Continuum/ContinuumBenchmarks/MNIST/Continual-Learning-Benchmark/scripts/outputs/permuted_MNIST_incremental_domain_10"

def get_avg_acc(acc_dict, num_tasks):


    acc_matrix = to_tensor(acc_dict, num_tasks)
    avg_acc = torch.zeros(num_tasks)
    task_avg_acc = torch.zeros(num_tasks)
    # Average Acc
    for col_index in range(0, num_tasks):
        avg_acc[col_index] = torch.sum(acc_matrix[:, col_index]) / (col_index + 1)
        task_avg_acc[col_index] = torch.sum(acc_matrix[col_index:(col_index + 1), col_index:]) / (num_tasks - col_index)
    print("Avg_acc:", avg_acc)
    print("Task Avg_acc:", task_avg_acc)


    return avg_acc, task_avg_acc


def to_tensor(acc_dict, num_tasks):
    # num_tasks = len(acc_dict.keys())
    acc_matrix = torch.zeros(size=(num_tasks, num_tasks))
    for row_key in acc_dict.keys():
        for col_key in acc_dict[row_key].keys():
            acc_matrix[int(row_key) - 1][int(col_key) - 1] = acc_dict[row_key][col_key]

    return acc_matrix
if __name__ == '__main__':

    for approach in Approaches:
        avg_acc_lst = []
        task_avg_acc_lst = []
        for exp_rep in range(1, REPEAT+1):
            acc_matrix_path = "{0}/{1}_{2}-precision_record.pt".format(OutDirPath, approach, exp_rep)
            print(acc_matrix_path)
            acc_dict = torch.load(acc_matrix_path)
            num_tasks = len(acc_dict.keys())
            avg_acc, task_avg_acc = get_avg_acc(acc_dict, num_tasks)
            avg_acc_lst.append(avg_acc.numpy())
            task_avg_acc_lst.append(task_avg_acc.numpy())
        avg_acc_pd = pd.DataFrame(avg_acc_lst, columns=list(range(1, num_tasks + 1)))
        avg_acc_pd.to_csv("{0}/_PD_{1}.csv".format(OutDirPath, approach), index=False)
        # break
    # print(avg_acc_lst)

    print(avg_acc_pd)
    # for key in acc_dict.keys():
    #     print("K:", key)
    # print(acc_dict)
    # print(acc_dict["1"])

    # print("Avg Acc:", torch.mean(acc_matrix, dim=0))
    # print("Task Avg:", torch.mean(acc_matrix, dim=1))
    # print(acc_matrix)
    # acc_matrix_zeros = torch.zeros(size=(10, 10))
    # acc_matrix_ones = torch.ones(size=(10, 10))
    # acc_matrix_zeros[0][0] = 100

    # avg = acc_matrix_zeros + acc_matrix_ones
    # print(avg/2)
