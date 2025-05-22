"""
Módulo de compatibilidade para redirecionamento para local_storage.py.
Este módulo permite que o código que importa de cloud_storage.py continue funcionando.
"""

# Importar e expor funções do local_storage.py
from ..preprocessing.local_storage import (
    connect_to_gcs,
    list_files_by_extension,
    categorize_files,
    extract_launch_id,
    load_csv_or_excel,
    load_csv_with_auto_delimiter
)

# Nota: A classe storage não é necessária, pois connect_to_gcs não a utiliza mais