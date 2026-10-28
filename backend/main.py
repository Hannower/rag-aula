import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import settings
from backend.database import db
from backend.rag_service import rag_service

app = FastAPI(
    title="RAG Didático API - Teophilo",
    description="API para demonstração visual e didática de um pipeline RAG completo",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    pergunta: str
    top_k: int = Field(default=5, ge=1, le=20)
    min_threshold: float = Field(default=0.35, ge=0, le=1)

class IngestRequest(BaseModel):
    filename: Optional[str] = None

@app.get("/api/status")
def get_status():
    supabase_ok = db.check_connection()
    openrouter_configured = bool(settings.OPENROUTER_API_KEY)
    
    return {
        "status": "online",
        "services": {
            "supabase": "conectado" if supabase_ok else "falha_conexao",
            "openrouter": "configurado" if openrouter_configured else "nao_configurado",
            "fastembed_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION
        }
    }

@app.get("/api/available-files")
def list_available_files():
    """Mostra os PDFs que ainda estão na pasta raw."""
    try:
        summary = db.get_ingestion_summary()
    except Exception:
        raise HTTPException(status_code=503, detail="Não foi possível consultar a ingestão no Supabase.")

    local_pdfs = [
        name for name in os.listdir(settings.DOCS_DIR)
        if name.lower().endswith(".pdf")
        and os.path.isfile(os.path.join(settings.DOCS_DIR, name))
    ] if os.path.exists(settings.DOCS_DIR) else []

    return {
        "files": sorted(local_pdfs),
        "total_arquivos_locais": len(local_pdfs),
        "total_documentos_vetorizados": summary["total_documentos_vetorizados"],
        "total_chunks_vetorizados": summary["total_chunks_vetorizados"],
    }

@app.get("/api/documents")
def list_documents():
    docs = db.list_documentos()
    return {"documentos": docs}

@app.post("/api/ingest")
def ingest_documents(req: IngestRequest):
    # Descobre o que já foi vetorizado antes de iniciar qualquer ingestão.
    try:
        summary = db.get_ingestion_summary()
    except Exception:
        logging.getLogger(__name__).exception("Falha ao verificar arquivos já indexados antes da ingestão.")
        raise HTTPException(status_code=503, detail="Não foi possível consultar a ingestão no Supabase.")

    indexed_names = set(summary["arquivos_indexados"])
    if req.filename:
        # Aceita apenas um nome de arquivo simples dentro da pasta documentos/raw.
        if os.path.basename(req.filename) != req.filename or not req.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Informe apenas o nome de um arquivo PDF da pasta documentos/raw.")
        if not os.path.isfile(os.path.join(settings.DOCS_DIR, req.filename)):
            raise HTTPException(status_code=404, detail="O PDF solicitado não foi encontrado na pasta documentos/raw.")
        files_to_process = [req.filename]
    else:
        if os.path.exists(settings.DOCS_DIR):
            files_to_process = [
                name for name in os.listdir(settings.DOCS_DIR)
                if name.lower().endswith(".pdf")
                and os.path.isfile(os.path.join(settings.DOCS_DIR, name))
            ]
        else:
            files_to_process = []
    
    if not files_to_process:
        raise HTTPException(status_code=400, detail="Nenhum arquivo PDF encontrado para ingestão.")

    results = []
    archived_files = []
    processing_attempts = 0
    for fn in files_to_process:
        if fn in indexed_names:
            try:
                archived_path = rag_service.archive_ingested_document(fn)
                archived_files.append(os.path.basename(archived_path))
            except Exception as e:
                results.append({
                    "nome_arquivo": fn,
                    "status": "ERRO",
                    "erro": str(e)
                })
            continue

        processing_attempts += 1
        try:
            res = rag_service.process_and_ingest_document(fn)
            results.append(res)
        except Exception as e:
            results.append({
                "nome_arquivo": fn,
                "status": "ERRO",
                "erro": str(e)
            })

    summary_updated = True
    warnings = []
    try:
        updated_summary = db.get_ingestion_summary()
    except Exception:
        logging.getLogger(__name__).exception("Falha ao atualizar resumo após a ingestão.")
        # O resumo é informativo: sua falha não desfaz os arquivos já processados.
        updated_summary = summary.copy()
        summary_updated = False
        warnings.append(
            "Não foi possível atualizar o resumo no Supabase. Os resultados por arquivo "
            "foram preservados; as contagens e arquivos_indexados são anteriores à ingestão."
        )
    inserted_files = [
        result["nome_arquivo"]
        for result in results
        if result.get("status") == "CONCLUIDO"
    ]
    failed_files = [
        result["nome_arquivo"]
        for result in results
        if result.get("status") == "ERRO"
    ]

    if inserted_files:
        message = (
            "Ingestão concluída com sucesso no Supabase para: "
            f"{', '.join(inserted_files)}."
        )
        if failed_files:
            message += f" Falha na ingestão de: {', '.join(failed_files)}."
    elif archived_files:
        message = (
            "Os documentos já estavam indexados e foram movidos para gold: "
            f"{', '.join(archived_files)}."
        )
    else:
        message = "Nenhum documento foi inserido com sucesso."

    return {
        "total_processados": processing_attempts,
        "detalhes": results,
        "mensagem": message,
        "arquivos_inseridos": inserted_files,
        "resumo_atualizado": summary_updated,
        "avisos": warnings,
        "arquivos_movidos_para_gold": archived_files,
        **updated_summary,
    }

@app.post("/api/query")
def execute_query(req: QueryRequest):
    if not req.pergunta.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")
    
    try:
        resultado = rag_service.query_rag(
            question=req.pergunta,
            top_k=req.top_k,
            min_threshold=req.min_threshold
        )
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar consulta: {str(e)}")

@app.delete("/api/documents")
def clear_documents():
    try:
        db.clear_all()
        return {"mensagem": "Todos os documentos e chunks foram removidos com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao limpar banco: {str(e)}")
