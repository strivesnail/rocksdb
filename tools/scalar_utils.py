"""共享的 scaler 工具，供训练脚本和 ml_predict 使用"""


class NoScaler:
    """不做任何缩放，直接返回原数据。与 sklearn scaler 接口兼容。"""

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X
