import argparse 
import os 
import re 
import time
import hashlib 
import csv 
from urllib.parse import urljoin, urlparse 
from collections import deque

import requests 
from bs4 import BeautifulSoup 
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from typing import cast
import pandas as pd

USER_AGENT = "DatasetScraper/1.0 (+https://yourproject.example)"

def sane_filename(s: str) -> str:  # Remove caracteres problemáticos para nomes de arquivo
    s = re.sub(r"[\/:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s.strip("_")

def is_same_domain(base: str, url: str) -> bool: 
    return urlparse(base).netloc == urlparse(url).netloc

def get_soup(session: requests.Session, url: str, timeout: int = 15): 
    resp = session.get(url, timeout=timeout) 
    resp.raise_for_status() 
    return BeautifulSoup(resp.text, "html.parser")

def image_is_valid(response: requests.Response) -> bool: 
    ctype: str = response.headers.get("Content-Type", "") 
    return ctype.startswith("image/")

def sha1_bytes(b: bytes) -> str: 
    return hashlib.sha1(b).hexdigest()

def ensure_dir(path: str) -> None: 
    if not os.path.exists(path): 
        os.makedirs(path, exist_ok=True)

def crawl_site(base_url: str, session: requests.Session, max_pages: int = 500, depth: int = 2, delay: float = 1.0, verbose: bool = False):
    visited: set[str] = set()  # URLs já visitadas
    pages: list[str] = []
    queue = deque([(base_url, 0)])  # (URL, profundidade atual)

    while queue and len(visited) < max_pages:
        url, current_depth = queue.popleft()
        if url in visited:
            continue
        try:
            if verbose:
                print(f"Crawling: {url} (depth {current_depth})")
            soup = get_soup(session, url)     
        except Exception as e:
            if verbose:
                print(f"Erro ao acessar {url}: {e}")
            continue
        visited.add(url)
        pages.append(url)
        if current_depth < depth:
            for a in soup.find_all("a"):
                href_raw = a.get("href")
                if href_raw is None or not isinstance(href_raw, str):
                    continue
                href =href_raw
                full = urljoin(url, href)
                full = full.split('#')[0]  # remover fragmentos
                if is_same_domain(base_url, full) and full not in visited:
                    queue.append((full, current_depth + 1))
        time.sleep(delay)

    return pages

def extract_images_from_page(session: requests.Session, page_url: str, base_url: str): 
    try: 
        soup = get_soup(session, page_url) 
    except Exception as e: 
        print(f"Erro ao parsear {page_url}: {e}") 
        return []
    imgs = []
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        img_url = urljoin(page_url, src)
    # filtrar images que não pertençam ao domínio? mantemos qualquer imagem válida
        alt = img.get("alt", "")
        imgs.append((img_url, alt, str(img)))
    return imgs

def download_image(session, img_url, referer=None, timeout=20, max_retries=2): 
    headers = {"User-Agent": USER_AGENT} 
    if referer: 
        headers["Referer"] = referer
    for attempt in range(max_retries+1):
        try:
            resp = session.get(img_url, headers=headers, timeout=timeout, stream=True)
            if resp.status_code == 200 and image_is_valid(resp):
                content = resp.content
                return content, resp.headers.get("Content-Type")
            else:
                return None, None
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1 + attempt*1.5)
                continue
            else:
                print(f"Erro baixando {img_url}: {e}")
                return None, None

def save_image_bytes(b: bytes, out_path: str): 
    with open(out_path, "wb") as f: f.write(b)

def get_image_size_from_bytes(b: bytes): 
    try: 
        im = Image.open(BytesIO(b)) 
        return im.size  # (width, height) 
    except Exception: 
        return None

def main(args): 
    session = requests.Session() 
    session.headers.update({"User-Agent": USER_AGENT})
    base_url = args.base_url
    output = args.output
    ensure_dir(output)

# arquivo de metadados (resume-friendly)
meta_csv = os.path.join(output, "metadata.csv")
    if os.path.exists(meta_csv):
        df_meta = pd.read_csv(meta_csv)
        downloaded_hashes = set(df_meta["sha1"].dropna().astype(str).tolist())
    else:
        df_meta = pd.DataFrame(columns=["page_url", "img_url", "local_path", "sha1", "width", "height"])
        downloaded_hashes = set()

    # 1) Fazer crawling para descobrir páginas internas
    print("[*] Iniciando crawling para descobrir páginas (isso pode demorar um pouco)...")
    pages = crawl_site(base_url, session, max_pages=args.max_pages, depth=args.depth, delay=args.delay, verbose=args.verbose)

    print(f"[*] Páginas encontradas: {len(pages)}")

    # filtrar páginas que provavelmente contenham imagens de rochas: heurística simples
    candidate_pages = [
        p for p in pages 
        if re.search(r"rock|granite|basalt|gabbro|andesite|dolerite|porphy|thin|section|samples|petro", p, re.IGNORECASE) 
        or p.startswith(base_url)]
        # garantir que o base_url esteja na lista
    if base_url not in candidate_pages:
    candidate_pages.insert(0, base_url)

    print(f"[*] Páginas candidatas após heurística: {len(candidate_pages)}")

    # 2) Para cada página candidata, extrair imagens
    rows = []
    for page in tqdm(candidate_pages, desc="Páginas"):
        try:
            imgs = extract_images_from_page(session, page, base_url)
        except Exception as e:
            print(f"Erro extraindo imagens de {page}: {e}")
            continue

        # categoria: usar o título da página (se disponível) ou slug
        try:
            soup = get_soup(session, page)
            title_tag = soup.find("title")
            if title_tag and title_tag.text.strip():
                categoria = sane_filename(title_tag.text.strip())
            else:
                categoria = sane_filename(urlparse(page).path.strip('/').replace('/', '_') or 'root')
        except Exception:
            categoria = sane_filename(urlparse(page).path.strip('/').replace('/', '_') or 'root')

        pasta_cat = os.path.join(output, categoria)
        ensure_dir(pasta_cat)

        for img_url, alt, img_tag in imgs:
            # pequeno filtro: extensão conhecida
            if not re.search(r"\.(jpg|jpeg|png|gif|tif|tiff)$", img_url, re.IGNORECASE):
                # ainda assim tentamos se o content-type for imagem
                pass

                # download
            b, ctype = download_image(session, img_url, referer=page, timeout=args.timeout, max_retries=args.retries)
            if not b:
                continue

            h = sha1_bytes(b)
            if h in downloaded_hashes:
                # já baixado - pular
                continue

            # determinar extensão a partir do content-type
            ext = None
            if ctype:
                m = re.search(r"image/([a-zA-Z0-9+.-]+)", ctype)
                if m:
                    sub = m.group(1)
                    if sub == 'jpeg':
                        ext = 'jpg'
                    else:
                        ext = sub
            if not ext:
                # fallback do URL
                parsed_ext = os.path.splitext(img_url.split('?')[0])[1].lower().strip('.')
                ext = parsed_ext if parsed_ext else 'jpg'

            filename = f"{h}.{ext}"
            out_path = os.path.join(pasta_cat, filename)

            try:
                save_image_bytes(b, out_path)
                size = get_image_size_from_bytes(b) or (None, None)
                downloaded_hashes.add(h)

                row = {
                    "page_url": page,
                    "img_url": img_url,
                    "local_path": out_path,
                    "sha1": h,
                    "width": size[0],
                    "height": size[1]
                }
                rows.append(row)
            except Exception as e:
                print(f"Erro salvando {img_url} -> {out_path}: {e}")

            time.sleep(args.delay)

            # salvar metadados (append ao existente)
    if rows:
        df_new = pd.DataFrame(rows)
        df_meta = pd.concat([df_meta, df_new], ignore_index=True)
        df_meta.drop_duplicates(subset=["sha1"], inplace=True)
        df_meta.to_csv(meta_csv, index=False)

    print(f"[*] Concluído. Metadados salvos em: {meta_csv}")

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Scraper automático de imagens - Alex Strekeisen archive") 
    parser.add_argument('--base-url', required=True, help='URL base do site (ex: https://www.alexstrekeisen.it/english/')
    parser.add_argument('--output', default='dataset', help='Pasta de saída para imagens e metadata.csv') 
    parser.add_argument('--delay', type=float, default=1.5, help='Delay entre requisições (segundos)') 
    parser.add_argument('--max-pages', type=int, default=500, help='Número máximo de páginas a visitar no crawl') 
    parser.add_argument('--depth', type=int, default=2, help='Profundidade do crawl (BFS)') 
    parser.add_argument('--timeout', type=int, default=20, help='Timeout para download de imagens (segundos)') 
    parser.add_argument('--retries', type=int, default=2, help='Número de tentativas de download de imagem') 
    parser.add_argument('--verbose', action='store_true', help='Mostra mensagens extras durante o crawling')

    args = parser.parse_args()
    main(args)