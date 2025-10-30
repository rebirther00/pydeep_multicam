import numpy as np

class 퍼셉트론:
    def __init__(self):
        self.w = None
        self.b = None
        
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y = np.where(z > 0, 1, -1)
        return y
    
    def fit(self, X, y, 학습률, 학습횟수, 매개변수_출력=True):
        """학습 알고리즘"""
        # 1) 매개변수 초기화
        self.w = np.zeros(X.shape[-1])
        self.b = 0.0
        # 2) 매개변수 갱신
        for i in range(학습횟수):
            if 매개변수_출력:
                print(f'[{i}] w={self.w}, b={self.b}')
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                error = yi - y_pred
                갱신 = error * 학습률
                self.w += 갱신 * xi
                self.b += 갱신