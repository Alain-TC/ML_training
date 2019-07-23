from sklearn.pipeline import make_pipeline
import sklearn


def pipe_feature_engineering(steps):
    return sklearn.pipeline.Pipeline(steps)
