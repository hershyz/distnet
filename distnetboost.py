import distnet

def optimized_model(df, output):
    
    features = df.feature_map
    
    accuracy_map = {}
    for feature in features:
        if feature != output:
            curr_model = distnet.train(df, [feature], output)
            accuracy_map[feature] = distnet.eval(curr_model, df)
    accuracy_map = sorted(accuracy_map.items(), key=lambda x:x[1], reverse=True)

    res_features = []
    res_acc = 0

    for feature in accuracy_map:
        temp_features = []
        for existing in res_features:
            temp_features.append(existing)
        temp_features.append(feature[0])
        curr_model = distnet.train(df, temp_features, output)
        curr_acc = distnet.eval(curr_model, df)
        if curr_acc > res_acc:
            res_acc = curr_acc
            res_features.append(feature[0])
        else:
            break
    
    return distnet.train(df, res_features, output)