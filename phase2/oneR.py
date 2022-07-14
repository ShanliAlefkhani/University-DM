from files import get_dataset
from sklearn.model_selection import train_test_split
import numpy as np


def oneR_firstStep(data):
    temp_d = []
    start = 0
    end = 0
    while end < len(data):
        c = data[start][1]
        end = start
        for d in data[start:]:
            if d[1] != c:
                temp_d.append([start, end-1])
                start = end
                break
            end += 1
    return temp_d 

def oneR_secondStep(oner_fd, data, threshold):
    temp_d = []
    while len(oner_fd) > 1:
        zero_count = 0
        one_count = 0
        start = 0
        end = 0
        temp = [-1, 0, 0, 0, 0]
        while zero_count < threshold and one_count < threshold:
            start = oner_fd[0][0]
            if temp[0] == -1:
                temp[0] = start
            end = oner_fd[0][1]
            label = data[start][1]
            if label == 1:
                one_count += end - start
            else:
                zero_count += end - start
            del oner_fd[0]
            if len(oner_fd) == 0:
                break
            
        temp[1] = end
        temp[2] = zero_count
        temp[3] = one_count
        temp_d.append(temp)
    return temp_d

def oneR_thirdStep(oner_ds, data):

    def max_label(index):
        label = max(oner_ds[index][2], oner_ds[index][3])
        index_label = oner_ds[index].index(label)
        if index_label == 2:
            label = 0
        else:
            label = 1

        return label

    temp_d = []
    index = 0
    while index < len(oner_ds)-1:
        label = max_label(index)
        temp = oner_ds[index]
        for i in range(index+1, len(oner_ds)):
            temp_l = max_label(i)
            index = i
            if label == temp_l:
                temp[1] = oner_ds[i][1]
                temp[2] += oner_ds[i][2]
                temp[3] += oner_ds[i][3]
            else:
                break
        temp[4] = data[temp[1]][0]
        temp.append(label)
        temp_d.append(temp)
    
    return temp_d

def cal_error(oner_dt):
    error_sum = 0
    total = 0
    for d in oner_dt:
        error_sum += min(d[2], d[3])
        total += d[2] + d[3]

    return error_sum / total

def predict(min_error, x):
    feature = min_error[0]
    for i in min_error[2]:
        if x[feature] < i[4]:
            return i[5]    

df = get_dataset()
y = df[["cardio"]]
df_num = df.select_dtypes(exclude='object')
df_num_cat = df_num.drop(["smoke", "alco", "active"], axis=1)
df_temp1 = df_num_cat.drop("cardio", axis=1)
error_rate_sum = 0
print(df_num_cat)
for _ in range(5):
    X_train, X_test, y_train, y_test = train_test_split(df_temp1, y, test_size=0.2, random_state=1000)
    print(X_train)
    df = X_train.join(y_train)

    # data_top = data.head()

    errors = []

    df_temp = df[["age", "cardio"]]
    df_temp = df_temp.sort_values("age")
    #print(df_temp)
    #print(df_temp["cardio"].value_counts())
    df_temp = df_temp.to_numpy()
    oner_f = oneR_firstStep(df_temp)
    #print(oner_f)
    oner_s = oneR_secondStep(oner_f, df_temp, len(df_temp) // 10)
    #print(oner_s)
    age_oner_t = oneR_thirdStep(oner_s, df_temp)
    print(age_oner_t)
    errors.append([0, cal_error(age_oner_t), age_oner_t])

    df_temp = df[["height", "cardio"]]
    df_temp = df_temp.sort_values("height")
    print(df_temp["cardio"].value_counts())
    df_temp = df_temp.to_numpy()
    oner_f = oneR_firstStep(df_temp)
    oner_s = oneR_secondStep(oner_f, df_temp, len(df_temp) // 10)
    height_oner_t = oneR_thirdStep(oner_s, df_temp)
    print(height_oner_t)
    errors.append([1, cal_error(height_oner_t), height_oner_t])

    df_temp = df[["weight", "cardio"]]
    df_temp = df_temp.sort_values("weight")
    print(df_temp["cardio"].value_counts())
    df_temp = df_temp.to_numpy()
    oner_f = oneR_firstStep(df_temp)
    oner_s = oneR_secondStep(oner_f, df_temp, len(df_temp) // 10)
    weight_oner_t = oneR_thirdStep(oner_s, df_temp)
    print(weight_oner_t)
    errors.append([2, cal_error(weight_oner_t), weight_oner_t])

    df_temp = df[["ap_hi", "cardio"]]
    df_temp = df_temp.sort_values("ap_hi")
    print(df_temp["cardio"].value_counts())
    df_temp = df_temp.to_numpy()
    oner_f = oneR_firstStep(df_temp)
    oner_s = oneR_secondStep(oner_f, df_temp, len(df_temp) // 10)
    aphi_oner_t = oneR_thirdStep(oner_s, df_temp)
    print(aphi_oner_t)
    errors.append([3, cal_error(aphi_oner_t), aphi_oner_t])

    df_temp = df[["ap_lo", "cardio"]]
    df_temp = df_temp.sort_values("ap_lo")
    print(df_temp["cardio"].value_counts())
    df_temp = df_temp.to_numpy()
    oner_f = oneR_firstStep(df_temp)
    oner_s = oneR_secondStep(oner_f, df_temp, len(df_temp) // 10)
    aplo_oner_t = oneR_thirdStep(oner_s, df_temp)
    print(aplo_oner_t)
    errors.append([4, cal_error(aplo_oner_t), aplo_oner_t])

    min_error = min(errors, key=lambda x: x[1])
    print("min error:", min_error)

    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    y_test = np.ravel(y_test)


    error_rate = 0
    for i in range(len(X_test)):
        #print("predicted:", predict(min_error, X_test[i]), "actual:", y_test[i])
        if predict(min_error, X_test[i]) != y_test[i]:
            error_rate += 1
    error_rate /= len(y_test)
    error_rate_sum += error_rate
    print("error_rate:", error_rate)
print("avarage error rate:", error_rate_sum/5)