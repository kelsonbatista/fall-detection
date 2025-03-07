from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2

# ⚙️ Configuração do dispositivo (GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 1️⃣ Treinar o modelo YOLOv8
model_yolo = YOLO('yolov8n.pt')  # YOLOv8 Nano para velocidade
model_yolo.train(data='path/to/dataset.yaml', epochs=50, imgsz=640)

# 🔹 2️⃣ Função para extrair características (bounding boxes)
def extract_features(results):
    features = []
    for r in results:
        if hasattr(r, "boxes"):  # Verifica se existem caixas detectadas
            for box in r.boxes.xywh.cpu().numpy():  # Converte para numpy
                x_center, y_center, width, height = box
                features.append([x_center, y_center, width, height])
    return np.array(features) if features else np.zeros((1, 4))  # Evita erro se não houver caixas

# 🔹 3️⃣ Definição do modelo LSTM
class FallLSTM(nn.Module):
    def __init__(self):
        super(FallLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c_0 = torch.zeros(2, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# 🔹 4️⃣ Criar e configurar o modelo LSTM
model_lstm = FallLSTM().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=0.001)

# 🔹 5️⃣ Treinar o LSTM
# Supondo que você já tenha X_train (sequências de bounding boxes) e y_train (rótulos)
X_train = torch.rand((500, 10, 4)).to(device)  # Simulação de dados (500 amostras, 10 frames cada, 4 features)
y_train = torch.randint(0, 2, (500,)).float().to(device)  # Simulação de rótulos (queda ou não)

for epoch in range(50):
    model_lstm.train()
    optimizer.zero_grad()
    output = model_lstm(X_train).squeeze()
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 🔹 6️⃣ Salvar o modelo treinado
torch.save(model_lstm.state_dict(), "fall_lstm.pth")

# 🔹 7️⃣ Carregar o modelo para inferência
model_lstm.load_state_dict(torch.load("fall_lstm.pth", map_location=device))
model_lstm.eval()

# 🔹 8️⃣ Avaliação em tempo real (detecção de quedas)
cap = cv2.VideoCapture('test_video.mp4')
sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🔍 Previsão do YOLOv8
    results = model_yolo.predict(frame, conf=0.5)
    features = extract_features(results)
    sequence.append(features)

    if len(sequence) == 10:  # Janela temporal de 10 frames
        with torch.no_grad():  # Desativa gradientes para inferência
            input_seq = torch.tensor([sequence], dtype=torch.float32).to(device)
            pred = model_lstm(input_seq)

        if pred.item() > 0.5:
            print("🚨 Queda detectada!")

        sequence.pop(0)  # Remove o primeiro frame para manter a sequência deslizante

cap.release()
cv2.destroyAllWindows()
