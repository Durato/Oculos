import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import time
import queue
import threading
import cv2
import pyttsx3
from ultralytics import YOLO

MODELO = "yolov8n.pt"

fila_fala = queue.Queue(maxsize=1)

def trabalhador_voz():
    while True:
        texto = fila_fala.get()
        if texto is None:
            break
        try:
            voz = pyttsx3.init()
            voz.setProperty("rate", 170)
            voz.say(texto)
            voz.runAndWait()
            voz.stop()
            del voz
        except Exception as e:
            print(f"Erro na fala: {e}")

NOMES_PT = {
    "person": "pessoa", "bicycle": "bicicleta", "car": "carro",
    "motorcycle": "moto", "airplane": "avião", "bus": "ônibus",
    "train": "trem", "truck": "caminhão", "boat": "barco",
    "traffic light": "semáforo", "fire hydrant": "hidrante",
    "stop sign": "placa de pare", "parking meter": "parquímetro",
    "bench": "banco", "bird": "pássaro", "cat": "gato", "dog": "cachorro",
    "horse": "cavalo", "sheep": "ovelha", "cow": "vaca", "elephant": "elefante",
    "bear": "urso", "zebra": "zebra", "giraffe": "girafa",
    "backpack": "mochila", "umbrella": "guarda-chuva", "handbag": "bolsa",
    "tie": "gravata", "suitcase": "mala", "frisbee": "frisbee",
    "skis": "esquis", "snowboard": "snowboard", "sports ball": "bola",
    "kite": "pipa", "baseball bat": "taco de beisebol",
    "baseball glove": "luva de beisebol", "skateboard": "skate",
    "surfboard": "prancha de surf", "tennis racket": "raquete de tênis",
    "bottle": "garrafa", "wine glass": "taça", "cup": "copo",
    "fork": "garfo", "knife": "faca", "spoon": "colher", "bowl": "tigela",
    "banana": "banana", "apple": "maçã", "sandwich": "sanduíche",
    "orange": "laranja", "broccoli": "brócolis", "carrot": "cenoura",
    "hot dog": "cachorro-quente", "pizza": "pizza", "donut": "rosquinha",
    "cake": "bolo", "chair": "cadeira", "couch": "sofá",
    "potted plant": "vaso de planta", "bed": "cama", "dining table": "mesa",
    "toilet": "vaso sanitário", "tv": "televisão", "laptop": "notebook",
    "mouse": "mouse", "remote": "controle remoto", "keyboard": "teclado",
    "cell phone": "celular", "microwave": "microondas", "oven": "forno",
    "toaster": "torradeira", "sink": "pia", "refrigerator": "geladeira",
    "book": "livro", "clock": "relógio", "vase": "vaso", "scissors": "tesoura",
    "teddy bear": "ursinho de pelúcia", "hair drier": "secador de cabelo",
    "toothbrush": "escova de dentes",
}

def traduzir(nome_en):
    chave = nome_en.lower()
    if chave in NOMES_PT:
        return NOMES_PT[chave]
    return chave

def main():
    print(f"Carregando modelo: {MODELO}")
    modelo = YOLO(MODELO)

    threading.Thread(target=trabalhador_voz, daemon=True).start()

    cam = None
    for indice in (0, 1, 2):
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
            tentativa = cv2.VideoCapture(indice, backend)
            if not tentativa.isOpened():
                tentativa.release()
                continue

            tentativa.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            for _ in range(5):
                tentativa.read()
                time.sleep(0.05)
            ok, frame_teste = tentativa.read()

            if ok and frame_teste is not None and float(frame_teste.mean()) > 5:
                cam = tentativa
                print(f"Webcam aberta (indice={indice}, backend={backend}, brilho={frame_teste.mean():.1f})")
                break
            print(f"Tentativa indice={indice} backend={backend} -> preto, descartando")
            tentativa.release()
        if cam is not None:
            break

    if cam is None:
        print("Não consegui abrir a webcam. Verifique se outro app está usando ela.")
        return

    print("Aquecendo a câmera...")
    for _ in range(15):
        cam.read()
        time.sleep(0.05)

    ultima_fala = 0
    intervalo_fala = 3.0

    print("Pressione Q para sair.")
    falhas = 0
    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            falhas += 1
            if falhas > 30:
                print("Muitas falhas seguidas, encerrando.")
                break
            continue
        falhas = 0

        resultados = modelo(frame, verbose=False, conf=0.5)[0]
        anotado = resultados.plot()
        cv2.imshow(f"Oculos - {MODELO}", anotado)

        nomes_detectados = set()
        for c in resultados.boxes.cls.tolist():
            nomes_detectados.add(modelo.names[int(c)])

        agora = time.time()
        if nomes_detectados and (agora - ultima_fala) > intervalo_fala:
            texto = "Estou vendo " + ", ".join(traduzir(n) for n in nomes_detectados)
            print(texto)
            if fila_fala.empty():
                fila_fala.put(texto)
            ultima_fala = agora

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
