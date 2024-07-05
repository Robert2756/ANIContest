import time
import numpy as np

BETA = 0.5

def criterion(feature, selected_features, I_xx, I_xy, beta):
    features_selected_sum = 0

    for selected_feature in selected_features:
        features_selected_sum += I_xx[selected_feature, feature]
    
    return I_xy[feature] - beta*features_selected_sum

def mutual_infxx(x1, x2):
    '''
    x -> input channel (3250)
    y -> output channel (3250)
    '''
    # Binwidth1 -> for each channel x1
    sum_squared_diff = 0
    mean_x1 = np.mean(x1, axis=0)

    for i in x1:
        sum_squared_diff += (i-mean_x1)**2

    sigma1 = (sum_squared_diff / (len(x1)-1))**(1/2)

    W1 = 3.5 * sigma1 *  1/(len(x1) ** (1 / 3))

    # Binwidth2 -> for each channel x2
    sum_squared_diff = 0
    mean_x2 = np.mean(x2, axis=0)

    for i in x2:
        sum_squared_diff += (i-mean_x2)**2

    sigma2 = (sum_squared_diff / (len(x2)-1))**(1/2)

    W2 = 3.5 * sigma2 *  1/(len(x2) ** (1 / 3))

    # Create histogramm of x1 and x2 (input, input)
    max1 = np.max(x1)
    min1 = np.min(x1)
    mean1 = np.mean(x1)

    max2 = np.max(x2)
    min2 = np.min(x2)
    mean2 = np.mean(x2)

    number_bins1 = int(np.ceil((np.max([(np.abs(max1)-mean1),(np.abs(min1)-mean1)])+W1/2)/W1)) * 2 + 1 # double of the bins plus the one in the middle
    number_bins2 = int(np.ceil((np.max([(np.abs(max2)-mean2),(np.abs(min2)-mean2)])+W2/2)/W2)) * 2 + 1 # double of the bins plus the one in the middle
    bins1 = []
    bins1.append([mean1-W1/2, mean1+W1/2])
    bins2 = []
    bins2.append([mean2-W2/2, mean2+W2/2])

    hist= np.zeros((number_bins1, number_bins2))

    indices1 = []
    indices2 = []

    # iterate x1
    for index, value in enumerate(x1):

        diff1 = mean1-value

        if diff1<0 or diff1==0: # x1 is bigger than mean
            if np.abs(diff1) > mean1-W1/2 and np.abs(diff1) < mean1+W1/2: # x is in the middle bin
                indices1.append(int(number_bins1/2))
            else:
                indices1.append(int(number_bins1/2) + 1 + int((np.abs(diff1)-W1/2)/W1)) # above the middle bin
        if diff1>0: # x is smaller than mean
            if diff1 > mean1-W1/2 and diff1 < mean1+W1/2: # x is in the middle bin
                indices1.append(int(number_bins1/2))
            else:
                indices1.append(int(number_bins1/2) - 1 - int((diff1-W1/2)/W1)) # beneath the middle bin

    # iterate x2
    for index, value in enumerate(x2):

        diff2 = mean2-value

        if diff2<0 or diff2==0: # x1 is bigger than mean
            if np.abs(diff2) > mean2-W2/2 and np.abs(diff2) < mean2+W2/2: # x is in the middle bin
                indices2.append(int(number_bins2/2))
            else:
                indices2.append(int(number_bins2/2) + 1 + int((np.abs(diff2)-W2/2)/W2)) # above the middle bin
        if diff2>0: # x is smaller than mean
            if diff2 > mean2-W2/2 and diff2 < mean2+W2/2: # x is in the middle bin
                indices2.append(int(number_bins2/2))
            else:
                indices2.append(int(number_bins2/2) - 1 - int((diff2-W2/2)/W2)) # beneath the middle bin

    for index, index1 in enumerate(indices1):
            hist[index1, indices2[index]] += 1

    prob_matrix = hist/3250

    # Calculate mutual information
    MI_x1_x2 = 0

    for column_index, column in enumerate(hist): # column
        for x_index, entry in enumerate(prob_matrix[column_index]): # row
            # sum all entries of the entry row
            px = 0
            for index, column in enumerate(hist):
                px += prob_matrix[index][x_index]

            # sum all entries of the y column
            py = 0
            for column_entry in prob_matrix[column_index]:
                py += column_entry

            if entry == 0:
                continue
            else:
                MI_x1_x2 += entry*np.log2(entry/(px*py))

    # print("MI_x1_x2: ", MI_x1_x2)
    return MI_x1_x2


def mutual_infxy(x, y):
    '''
    x -> input channel (3250)
    y -> output channel (3250)
    '''

    # Binwidth -> for each channel different
    sum_squared_diff = 0
    mean_x = np.mean(x, axis=0)

    for i in x:
        sum_squared_diff += (i-mean_x)**2

    sigma = (sum_squared_diff / (len(x)-1))**(1/2)

    W = 3.5 * sigma *  1/(len(x) ** (1 / 3))

    # Create histogramm of x and y (input, output)
    max = np.max(x)
    min = np.min(x)
    mean = np.mean(x)

    number_bins = int(np.ceil((np.max([(np.abs(max)-mean),(np.abs(min)-mean)])+W/2)/W)) * 2 + 1 # double of the bins plus the one in the middle
    bins = []
    bins.append([mean-W/2, mean+W/2])

    hist= np.zeros((2, number_bins))
    for index, value in enumerate(x):

        diff = mean-value
        row = y[index] # y=0 or y=1

        if diff<0 or diff==0: # x is bigger than mean
            if np.abs(diff) > mean-W/2 and np.abs(diff) < mean+W/2: # x is in the middle bin
                hist[row, int(number_bins/2)] += 1 # odd_number/2+1 -> center; center -1 -> index
            else:
                index = int(number_bins/2) + 1 + int((np.abs(diff)-W/2)/W) # above the middle bin
                hist[row, index] += 1
        if diff>0: # x is smaller than mean
            if diff > mean-W/2 and diff < mean+W/2: # x is in the middle bin
                hist[row, int(number_bins/2)] += 1
            else:
                index = int(number_bins/2) - 1 - int((diff-W/2)/W) # beneath the middle bin
                hist[row, index] += 1

    prob_matrix = hist/3250
    # print("prob matrix shape: ", np.array(prob_matrix).shape) # prob_matrix[0] -> y=0, prob_matrix[1] -> y=1

    # Calculate mutual information
    MI_x_y = 0

    for y in range(0, 2): # column
        for x_index, entry in enumerate(prob_matrix[y]): # row
            # sum all entries of the entry row
            px = prob_matrix[0][x_index] + prob_matrix[1][x_index]

            # sum all entries of the y column
            py = 0
            for column_entry in prob_matrix[y]:
                py += column_entry

            if entry == 0:
                continue
            else:
                MI_x_y += entry*np.log2(entry/(px*py))

    return MI_x_y

def MIFS(data, target, beta, max_feature_len):
    '''
    data -> shape(250, 3250)
    '''

    # MIFS algorithm
    selected_features = []
    remaining_features = list(range(data.shape[0])) # channel indices

    # Mutual information between input and target
    MI_input_target = []
    for channel in data:
        MI_input_target.append(mutual_infxy(channel, target)) # (2, 250)
    
    # Channel with the highest mutual information to target is starting point
    selected_features.append(np.argmax(MI_input_target)) # indices of MI_input_target
    remaining_features.remove(np.argmax(MI_input_target)) # indices of MI_input_target

    # Perform MIFS algorithm
    while len(selected_features)<max_feature_len:

        MIFS = []

        # Select channel with highest MIFS value from remaining ones
        for feature_index in remaining_features:
            # Compute MIFS value
            mutual_infxy_feature = mutual_infxy(data[feature_index], target)
            mutual_infxx_feature = 0
            for feature_index_selected in selected_features:
                mutual_infxx_feature += mutual_infxx(data[feature_index], data[feature_index_selected])
            MIFS.append(mutual_infxy_feature - beta*(mutual_infxx_feature))
            # print("MIFS: ", len(MIFS))

        # Channel with highest MIFS value is added to selected features
        selected_features.append(np.argmax(MIFS)) # indices of MI_input_target
        remaining_features.remove(np.argmax(MIFS)) # indices of MI_input_target
        print("Remaining features: ", remaining_features)
        print("Selected features: ", selected_features)
    
    return selected_features

