import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Lista de URLs das páginas
urls = [
    "https://www.alexstrekeisen.it/english/sedi/wackestone.php",
    "https://www.alexstrekeisen.it/english/sedi/oolite.php",
    "https://www.alexstrekeisen.it/english/sedi/mudstone.php",
    "https://www.alexstrekeisen.it/english/sedi/grainstone.php",
]
# Pasta principal
base_folder = "dataset"
os.makedirs(base_folder, exist_ok=True)

for url in urls:
    print(f"\n🔎 Acessando {url} ...")
    
    # Requisição da página
    response = requests.get(url)
    response.encoding = "utf-8"  # garantir acentos certos
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Título da página (limpando para nome de pasta)
    title = soup.title.string.strip() if soup.title else "sem_titulo"
    title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).rstrip()
    
    # Criar pasta da página
    page_folder = os.path.join(base_folder, title)
    os.makedirs(page_folder, exist_ok=True)
    print(f"📂 Criando pasta: {page_folder}")
    
    # Pegar todas as imagens "grandes"
    links = soup.find_all("a", onclick=True)
    
    for link in links:
        onclick = link.get("onclick", "")
        if "window.open" in onclick and ".jpg" in onclick:
            img_url = onclick.split("'")[1]  # pega a parte com a URL da imagem
            full_url = urljoin(url, img_url)
            
            # Nome do arquivo
            filename = os.path.basename(full_url)
            filepath = os.path.join(page_folder, filename)
            
            # Baixar imagem se ainda não existe
            if not os.path.exists(filepath):
                try:
                    img_data = requests.get(full_url).content
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                    print(f"✅ Baixada: {filename}")
                except Exception as e:
                    print(f"❌ Erro ao baixar {filename}: {e}")
            else:
                print(f"⏩ Já existe: {filename}")
