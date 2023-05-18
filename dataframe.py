class Dataframe:

    def __init__(self, csv):
        
        # read raw lines
        f = open(csv)
        raw = f.readlines()

        # strip newlines
        lines = []
        for i in range(len(raw)):
            lines.append(raw[i].replace('\n', ''))
        
        # convert lines arr to 2d point arr
        point_arr = []
        for i in range(1, len(lines)):
            point_arr.append(lines[i].split(','))
        self.point_arr = point_arr

        # extract features, map to respective index + type
        features = lines[0].split(',')
        feature_map = {}
        for i in range(len(features)):
            feature_test = point_arr[1][i]
            try:
                float(feature_test)
                feature_map[features[i]] = [i, 'numeric']
            except:
                feature_map[features[i]] = [i, 'categorical']
        self.feature_map = feature_map
    

    def display(self):
        print(self.feature_map)
        for row in self.point_arr:
            print(row)
    
    def get_test_points(self):
        
        reverse_feature_map = {}
        for feature in self.feature_map:
            reverse_feature_map[self.feature_map[feature][0]] = feature
        
        res = []
        for row in self.point_arr:
            curr = {}
            for i in range(len(row)):
                curr[reverse_feature_map[i]] = row[i]
            res.append(curr)
        
        return res