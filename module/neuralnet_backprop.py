import numpy as np

class Sigmoid:
    def __init__(self):
        self.y = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def backward(self, dout):
        dx = dout * self.y * (1 - self.y)
        return dx

class 완전연결:
    def __init__(self, 입력수, 출력수, 활성화=None):
        self.W = np.random.randn(입력수, 출력수)
        self.b = np.zeros(출력수)
        self.activation = 활성화
        self.x = None
        self.dW = None
        self.db = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        z = np.dot(x, self.W) + self.b
        if self.activation:
            return self.activation(z)
        return z

    def backward(self, dout):
        if self.activation:
            dout = self.activation.backward(dout)
            
        self.dW = np.dot(self.x.T, dout)
        # 배치 단위 연산 시, 미분값을 모든 표본에 대해 더합니다.
        self.db = np.sum(dout, axis=0)
        
        dx = np.dot(dout, self.W.T)
        return dx

def softmax(z):
    if z.ndim == 1:
        z = z.reshape(1, -1)
    exp_z = np.exp(z - np.max(z, axis=1).reshape(-1, 1))
    return exp_z / np.sum(exp_z, axis=1).reshape(-1, 1)

def 교차엔트로피오차(y, y_pred):
    delta = 1e-7
    배치크기 = y.shape[0]
    return -np.sum(y * np.log(y_pred + delta)) / 배치크기

class CrossEntropy:
    def __init__(self, from_logits=False):
        self.y = None
        self.from_logits = from_logits
        self.proba = None

    def __call__(self, z, y):
        return self.forward(z, y)

    def forward(self, outputs, y):
        self.y = y
        # z -> softmax -> proba
        if not self.from_logits:      
            self.proba = outputs
        else: # from_logits=True -> 실수값 -> 확률값
            self.proba = softmax(outputs)
        # CEE(y, proba)
        손실 = 교차엔트로피오차(y, self.proba)
        return 손실

    def backward(self, dout=1):
        배치크기 = len(self.y)
        dz = self.proba - self.y
        return dz / 배치크기

class 역전파신경망:
    def __init__(self, 손실함수):
        self.layers = []
        self.loss_func = 손실함수

    def add(self, layer):
        self.layers.append(layer)

    def __call__(self, x):
        """순전파"""
        outputs = x
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs # z_last

    def 손실산출(self, x, y):
        outputs = self(x)
        손실 = self.loss_func(outputs, y)
        return 손실

    def fit(self, x, y, 배치크기, 에폭수, 학습률):
        에폭당_배치수 = len(x) // 배치크기
        학습횟수 = 에폭당_배치수 * 에폭수
        print(f'배치크기={배치크기}, 에폭수={에폭수}, 학습횟수={학습횟수}({에폭당_배치수}/에폭)')
        손실변화 = []
        for 학습 in range(학습횟수):
            # 1. 미니 배치
            표본수 = S = len(x)
            배치색인 = np.random.choice(표본수, 배치크기)
            x_batch = x[배치색인]
            y_batch = y[배치색인]
            # 2. 경사 산출 (역전파)
            # 1) 순전파
            손실 = self.손실산출(x_batch, y_batch)
            손실변화.append(손실)
            # 2) 역전파            
            dout = self.loss_func.backward(1)
            for layer in reversed(self.layers):
                dout = layer.backward(dout)
            # 3. 매개변수 갱신 (경사 하강)
            for layer in self.layers:
                if isinstance(layer, 완전연결):
                    layer.W -= layer.dW * 학습률
                    layer.b -= layer.db * 학습률

            if 학습 == 0 or (학습 + 1) % 100 == 0:
                print(f'[학습 {학습 + 1}] Loss: {손실:.3f}')
        return 손실변화
