# distnet
Distribution-based inference model for ML classification problems.

Manual feature training:
```python
import dataframe
import distnet
import distnetboost

df = dataframe.Dataframe('test-data/Cancer_Data.csv')
model = distnet.train(df, ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'], 'diagnosis')
print(distnet.eval(model, df))
```

DistNetBoost algorithm -- prioritizes training on the fewest possible features while producing similar prediction accuracies:
```python
import dataframe
import distnet
import distnetboost

df = dataframe.Dataframe('test-data/drug200.csv')
model = distnetboost.optimized_model(df, 'Drug')
print(distnet.eval(model, df))
```
