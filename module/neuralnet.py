import numpy as np
import time

sigmoid = lambda z: 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - z.max()) # 오버플로 방지
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def 경사산출(f, x):
    h = 1e-4
    경사 = np.empty_like(x)
    
    for i, xi in enumerate(x):
        x[i] = xi + h
        fxh1 = f(x)
        x[i] = xi - h
        fxh2 = f(x)
        
        경사[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = xi # 원래 값 복원

    return 경사

def 경사산출_2d(f, X):
    경사 = np.zeros_like(X)
    
    for j, xj in enumerate(X):
        경사[j] = 경사산출(f, xj)
    return 경사

def 교차엔트로피오차(y, y_pred):
    delta = 1e-7
    배치크기 = y.shape[0]
    return -np.sum(y * np.log(y_pred + delta)) / 배치크기

class 다중퍼셉트론:
    def __init__(self, 입력수, 출력수, 활성화=None, seed=None):
        random = np.random.default_rng(seed)
        self.W = random.normal(size=(입력수, 출력수))
        self.b = np.zeros(shape=(출력수,))
        self.활성화 = 활성화

    def __call__(self, X):
        Z = X @ self.W + self.b # 교재 식[3.9]
        if self.활성화 is not None:
            return self.활성화(Z)
        return Z

class 신경망:
    def __init__(self, 손실함수):
        self.layers = []
        self.loss_func = 손실함수
        
    def add(self, layer):
        self.layers.append(layer)
        
    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def 손실산출(self, X, y):
        y_pred = self(X)
        손실 = self.loss_func(y, y_pred)
        return 손실
    
    def fit(self, X, y, 배치크기, 학습횟수, 학습률):
        """학습"""
        표본수 = X.shape[0]
        손실변화 = []
        for i in range(학습횟수):            
            print(f'학습 {i+1}/{학습횟수}')
            start_time = time.time()
            # 1. 미니배치
            배치색인 = np.random.choice(표본수, 배치크기)
            X_batch = X[배치색인]
            y_batch = y[배치색인]
            # 2. 경사 산출        
            f = lambda 매개변수: self.손실산출(X_batch, y_batch)
            층별경사 = []
            for layer in self.layers:
                dW = 경사산출_2d(f, layer.W)
                db = 경사산출(f, layer.b)
                층별경사.append((dW, db))
            # 3. 매개변수 갱신 (경사 하강)
            for layer, (dW, db) in zip(self.layers, 층별경사):
                layer.W -= dW * 학습률
                layer.b -= db * 학습률
            end_time = time.time()
            # (선택적) 손실확인
            손실 = self.손실산출(X_batch, y_batch)
            손실변화.append(손실)
            print(f'\t손실: {손실:.3f}, \t시간: {end_time - start_time:.1f}초')
        return 손실변화