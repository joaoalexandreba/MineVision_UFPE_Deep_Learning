from pathlib import Path
from roboflow import Roboflow
from src.config import API_KEY, DOWNLOAD_DIR, PROJECT, VERSION, WORKSPACE


def load_dataset(
    api_key: str = API_KEY,
    workspace: str = WORKSPACE,
    project: str = PROJECT,
    version: int = VERSION,
    download_dir: Path = DOWNLOAD_DIR,
) -> tuple[Path, Path]:
    """
    Baixa o dataset diretamente do Roboflow (ou reutiliza cache local)
    e retorna o diretório raiz e o caminho do data.yaml.

    Args:
        api_key: Chave da API do Roboflow.
        workspace: Nome da workspace no Roboflow.
        project: Nome do projeto.
        version: Versão do dataset.
        download_dir: Diretório de destino do download.

    Returns:
        tuple[Path, Path]: (raiz_dataset, data_yaml_path)
    """
    print("[1/6] Carregando dataset do Roboflow...")
    
    # Verificar se já foi baixado previamente
    data_yaml = None
    if download_dir.exists():
        for p in download_dir.rglob("data.yaml"):
            data_yaml = p
            break

    if data_yaml is None:
        rf = Roboflow(api_key=api_key)
        projeto = rf.workspace(workspace).project(project)
        versao = projeto.version(version)
        versao.download("yolov11", location=str(download_dir))

        for p in download_dir.rglob("data.yaml"):
            data_yaml = p
            break

    if data_yaml is None:
        raise FileNotFoundError(f"data.yaml não encontrado em {download_dir} após o download.")

    raiz_dataset = data_yaml.parent
    print(f"  Dataset localizado em: {raiz_dataset}")

    splits_encontrados = {}
    for split_name in ["train", "valid", "test", "val"]:
        img_dir = raiz_dataset / split_name / "images"
        if img_dir.exists():
            n = len(list(img_dir.glob("*.*")))
            splits_encontrados[split_name] = n
            print(f"  Split '{split_name}': {n} imagens")

    total = sum(splits_encontrados.values())
    print(f"  TOTAL de imagens disponíveis: {total}")
    return raiz_dataset, data_yaml
