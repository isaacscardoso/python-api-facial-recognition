from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np

app = FastAPI()

# Função para verificar se a imagem contém uma pessoa
def verifica_pessoa(imagem):
    # Carregar o classificador Haar Cascade para detecção de rosto
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Se faces for detectado, retornar True (contém uma pessoa), senão, False
    if len(faces) > 0:
        return True
    else:
        return False

@app.post("/verificar-pessoa/")
async def verificar_pessoa(imagem: UploadFile = File(...)):
    # Verificar se o arquivo enviado é uma imagem
    if not imagem.content_type.startswith('image'):
        raise HTTPException(status_code=415, detail="Apenas arquivos de imagem são suportados.")
    
    # Ler a imagem
    conteudo = await imagem.read()
    nparr = np.frombuffer(conteudo, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Verificar se a imagem contém uma pessoa
    if verifica_pessoa(img):
        return {"resultado": "A imagem contém uma pessoa."}
    else:
        return {"resultado": "A imagem não contém uma pessoa."}

@app.get("/testar-api-python/")
def hello_world_api():
    return {
        "status_code": 200,
        "message": "API Está Funcionando!"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)