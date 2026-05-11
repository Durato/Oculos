import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import time
import queue
import threading
import cv2
import pyttsx3
from ultralytics import YOLO

MODELO = "yolov8s-oiv7.pt"

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
    "person": "pessoa", "man": "homem", "woman": "mulher", "boy": "menino",
    "girl": "menina", "human face": "rosto", "human eye": "olho",
    "human head": "cabeça", "human hand": "mão", "human arm": "braço",
    "human leg": "perna", "human foot": "pé", "human body": "corpo",
    "human ear": "orelha", "human nose": "nariz", "human mouth": "boca",
    "human hair": "cabelo", "human beard": "barba",
    "bicycle": "bicicleta", "car": "carro", "motorcycle": "moto",
    "airplane": "avião", "aircraft": "avião", "bus": "ônibus", "train": "trem",
    "truck": "caminhão", "boat": "barco", "helicopter": "helicóptero",
    "van": "van", "taxi": "táxi", "ambulance": "ambulância",
    "traffic light": "semáforo", "traffic sign": "placa de trânsito",
    "fire hydrant": "hidrante", "stop sign": "placa de pare",
    "street light": "poste de luz", "bench": "banco",
    "bird": "pássaro", "cat": "gato", "dog": "cachorro", "horse": "cavalo",
    "sheep": "ovelha", "cow": "vaca", "pig": "porco", "rabbit": "coelho",
    "mouse": "rato", "hamster": "hamster", "squirrel": "esquilo",
    "fox": "raposa", "bear": "urso", "deer": "veado", "lion": "leão",
    "tiger": "tigre", "elephant": "elefante", "giraffe": "girafa",
    "zebra": "zebra", "monkey": "macaco", "snake": "cobra", "frog": "sapo",
    "turtle": "tartaruga", "lizard": "lagarto", "crocodile": "crocodilo",
    "goldfish": "peixe dourado", "fish": "peixe", "shark": "tubarão",
    "whale": "baleia", "dolphin": "golfinho", "octopus": "polvo",
    "crab": "caranguejo", "lobster": "lagosta", "shrimp": "camarão",
    "spider": "aranha", "scorpion": "escorpião", "bee": "abelha",
    "butterfly": "borboleta", "ladybug": "joaninha", "snail": "caracol",
    "ant": "formiga", "beetle": "besouro", "dragonfly": "libélula",
    "backpack": "mochila", "umbrella": "guarda-chuva", "handbag": "bolsa",
    "briefcase": "maleta", "tie": "gravata", "suitcase": "mala",
    "wallet": "carteira", "frisbee": "frisbee", "skis": "esquis",
    "snowboard": "snowboard", "sports ball": "bola", "ball": "bola",
    "football": "bola de futebol", "kite": "pipa",
    "baseball bat": "taco de beisebol", "baseball glove": "luva de beisebol",
    "skateboard": "skate", "surfboard": "prancha de surf",
    "tennis racket": "raquete de tênis",
    "bottle": "garrafa", "wine glass": "taça", "cup": "copo", "mug": "caneca",
    "fork": "garfo", "knife": "faca", "spoon": "colher", "bowl": "tigela",
    "plate": "prato", "jug": "jarro", "pitcher": "jarra",
    "banana": "banana", "apple": "maçã", "sandwich": "sanduíche",
    "orange": "laranja", "lemon": "limão", "strawberry": "morango",
    "grape": "uva", "watermelon": "melancia", "pineapple": "abacaxi",
    "mango": "manga", "peach": "pêssego", "pear": "pera",
    "tomato": "tomate", "broccoli": "brócolis", "carrot": "cenoura",
    "cabbage": "repolho", "cucumber": "pepino", "potato": "batata",
    "bread": "pão", "pasta": "macarrão", "rice": "arroz", "egg": "ovo",
    "cheese": "queijo", "hot dog": "cachorro-quente", "pizza": "pizza",
    "hamburger": "hambúrguer", "donut": "rosquinha", "doughnut": "rosquinha",
    "cookie": "biscoito", "cake": "bolo", "ice cream": "sorvete",
    "chocolate": "chocolate", "coffee cup": "xícara de café",
    "wine": "vinho", "beer": "cerveja",
    "chair": "cadeira", "couch": "sofá", "sofa bed": "sofá-cama",
    "potted plant": "vaso de planta", "houseplant": "planta",
    "plant": "planta", "flower": "flor", "tree": "árvore",
    "bed": "cama", "dining table": "mesa", "table": "mesa",
    "desk": "escrivaninha", "cabinet": "armário", "shelf": "prateleira",
    "wardrobe": "guarda-roupa", "drawer": "gaveta", "bookcase": "estante",
    "stool": "banquinho", "toilet": "vaso sanitário", "bathtub": "banheira",
    "shower": "chuveiro", "sink": "pia", "towel": "toalha",
    "tv": "televisão", "television": "televisão",
    "computer monitor": "monitor", "monitor": "monitor",
    "laptop": "notebook", "tablet computer": "tablet", "tablet": "tablet",
    "computer mouse": "mouse", "computer keyboard": "teclado",
    "keyboard": "teclado", "mobile phone": "celular", "cell phone": "celular",
    "telephone": "telefone", "remote control": "controle remoto",
    "remote": "controle remoto", "camera": "câmera", "headphones": "fones",
    "speaker": "caixa de som", "microphone": "microfone", "printer": "impressora",
    "calculator": "calculadora", "microwave": "microondas",
    "microwave oven": "microondas", "oven": "forno", "toaster": "torradeira",
    "refrigerator": "geladeira",
    "book": "livro", "notebook": "caderno", "pen": "caneta", "pencil": "lápis",
    "stapler": "grampeador", "scissors": "tesoura", "ruler": "régua",
    "glasses": "óculos", "sunglasses": "óculos de sol",
    "clock": "relógio", "wall clock": "relógio de parede",
    "watch": "relógio de pulso", "vase": "vaso", "mirror": "espelho",
    "candle": "vela", "lamp": "luminária", "light bulb": "lâmpada",
    "flag": "bandeira", "balloon": "balão", "box": "caixa", "can": "lata",
    "shirt": "camisa", "t-shirt": "camiseta", "pants": "calça",
    "jeans": "jeans", "dress": "vestido", "skirt": "saia", "shorts": "shorts",
    "shoe": "sapato", "sandal": "sandália", "boot": "bota",
    "hat": "chapéu", "cap": "boné", "helmet": "capacete", "coat": "casaco",
    "jacket": "jaqueta", "glove": "luva", "scarf": "cachecol", "sock": "meia",
    "hammer": "martelo", "wrench": "chave inglesa", "screwdriver": "chave de fenda",
    "drill": "furadeira", "saw": "serra", "axe": "machado",
    "teddy bear": "ursinho de pelúcia", "hair drier": "secador de cabelo",
    "hair dryer": "secador de cabelo", "toothbrush": "escova de dentes",
    "door": "porta", "window": "janela", "stairs": "escada",
    "building": "prédio", "house": "casa",
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
