class DistNet:

    def __init__(self, df, features, output):
        
        # init datastructures
        point_arr = df.point_arr
        feature_map = df.feature_map
        reverse_feature_map = {}
        for feature in feature_map:
            index = feature_map[feature][0]
            reverse_feature_map[index] = feature
        model_map = {}

        # find distinct outputs
        distinct_outputs = set()
        output_col = feature_map[output][0]
        for row in range(len(point_arr)):
            distinct_outputs.add(point_arr[row][output_col])

        # find min/max for every numeric output
        min_max = {}
        for feature in features:
            if feature_map[feature][1] == 'numeric' and feature != output:
                feature_index = feature_map[feature][0]
                min = float('inf')
                max = float('-inf')
                for row in point_arr:
                    curr = float(row[feature_index])
                    if curr < min:
                        min = curr
                    if curr > max:
                        max = curr
                min_max[feature] = [min, max]
        
        # construct numeric neurons
        divisions = 20
        for feature in min_max:
            neuron = {}
            curr = min_max[feature][0]
            step_size = (min_max[feature][1] - min_max[feature][0]) / divisions
            for i in range(divisions):
                key = [curr, curr + step_size]
                if i == 0:
                    key[0] = float('-inf')
                if i == divisions - 1:
                    key[1] = float('inf')
                neuron[tuple(key)] = {}
                curr += step_size
            model_map[feature] = neuron
        
        # construct categorical neurons
        for feature in features:
            if feature_map[feature][1] == 'categorical' and feature != output:
                model_map[feature] = {}
        
        # calculate percentage distributions
        n_points = len(point_arr)
        for row in point_arr:
            
            curr_output = row[feature_map[output][0]]

            for i in range(len(row)):
                
                feature = reverse_feature_map[i]
                if feature in features:
                    if feature_map[feature][1] == 'numeric' and feature != output:
                        value = float(row[i])
                        for key in model_map[feature]:
                            if value >= key[0] and value <= key[1]:
                                if curr_output in model_map[feature][key]:
                                    model_map[feature][key][curr_output] += 1 / n_points
                                else:
                                    model_map[feature][key][curr_output] = 1 / n_points
                                break
                    
                    if feature_map[feature][1] == 'categorical' and feature != output:
                        label = row[i]
                        if label not in model_map[feature]:
                            model_map[feature][label] = {}
                        if curr_output in model_map[feature][label]:
                            model_map[feature][label][curr_output] += 1 / n_points
                        else:
                            model_map[feature][label][curr_output] = 1 / n_points
        
        self.output_feature = output
        self.model_map = model_map
        self.distinct_outputs = distinct_outputs






def train(df, features, output):
    return DistNet(df, features, output)


def predict(point, model):
    
    model_map = model.model_map
    probabilities = {}
    for output in model.distinct_outputs:
        probabilities[output] = 1
    
    for feature in model_map:
        
        value = point[feature]

        try: # numerical
            value = float(value)
            for key in model_map[feature]:
                if value >= key[0] and value <= key[1]:
                    for output in probabilities:
                        if output not in model_map[feature][key]:
                            probabilities[output] = 0
                        else:
                            probabilities[output] *= model_map[feature][key][output]

        except: # not numerical
            label = value
            for output in probabilities:
                if output not in model_map[feature][label]:
                    probabilities[output] = 0
                else:
                    probabilities[output] *= model_map[feature][label][output]
        
    res = ''
    res_probability = -1
    for output in probabilities:
        if probabilities[output] > res_probability:
            res_probability = probabilities[output]
            res = output
    
    return res


def eval(model, df):

    output_feature = model.output_feature
    test_points = df.get_test_points()
    correct = 0

    for point in test_points:
        prediction = predict(point, model)
        real = point[output_feature]
        if prediction == real:
            correct += 1
    
    return (correct / len(test_points))