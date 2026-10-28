import io
import os
import shutil
import time
import uuid
from typing import List, Dict, Any, Optional, Union
import httpx
from backend.config import settings
from backend.database import db
from backend.extractor import extractor
from backend.chunker import chunker
from backend.tokenizer import tokenizer
from backend.embedder import embedder

class RAGService:
    @staticmethod
    def process_and_ingest_document(
        pdf_filename: str,
        file_bytes: Optional[bytes] = None,
        from_webhook: bool = False
    ) -> Dict[str, Any]:
        """
        Executa o pipeline completo de ingestão com observabilidade em cada fase.
        Suporta duas origens:
        - Pasta raw local (padrão): lê de settings.DOCS_DIR e move para settings.GOLD_DIR.
        - Webhook (from_webhook=True): recebe o binário diretamente em memória,
          não toca na pasta raw e grava diretamente na pasta gold após a indexação.
        """
        start_time = time.time()

        # =========================================================================
        # [CÓDIGO ADICIONADO - CONDIÇÃO 1: ORIGEM RAW vs WEBHOOK]
        # Se o arquivo vier via Webhook (from_webhook=True ou file_bytes fornecido),
        # ele NÃO é lido da pasta raw (settings.DOCS_DIR). O conteúdo binário em
        # memória é encapsulado em BytesIO para extração direta de texto.
        # Caso contrário (padrão), valida e carrega o arquivo físico da pasta raw.
        # =========================================================================
        if from_webhook or file_bytes is not None:
            if not file_bytes:
                raise ValueError("Para ingestão via webhook, é obrigatório fornecer o conteúdo binário (file_bytes).")
            pdf_input: Union[str, io.BytesIO] = io.BytesIO(file_bytes)
        else:
            file_path = os.path.join(settings.DOCS_DIR, pdf_filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Arquivo não encontrado na pasta raw: {pdf_filename}")
            pdf_input = file_path

        # 1. OCR / Extração de Texto
        t0 = time.time()
        raw_text = extractor.extract_text_from_pdf(pdf_input)
        parsed = extractor.parse_invoice_fields(raw_text, pdf_filename)
        extraction_time_ms = round((time.time() - t0) * 1000, 2)

        # 2. Persistir Documento no Supabase
        doc_payload = {
            "nome_arquivo": pdf_filename,
            "categoria": parsed["categoria"],
            "total_campos": parsed["total_campos"],
            "campos_preenchidos": parsed["campos_preenchidos"],
            "campos_nulos": parsed["campos_nulos"],
            "percentual_sucesso": parsed["percentual_sucesso"],
            "status": parsed["status"]
        }
        saved_doc = db.insert_documento(doc_payload)
        doc_id = saved_doc.get("id")
        if not doc_id:
            raise RuntimeError("O documento não foi salvo corretamente no Supabase.")

        # 3. Chunking
        t1 = time.time()
        raw_chunks = chunker.chunk_invoice(doc_id, parsed)
        chunking_time_ms = round((time.time() - t1) * 1000, 2)

        # 4. Tokenização Explícita
        t2 = time.time()
        tokenized_chunks = []
        texts_to_embed = []
        for chk in raw_chunks:
            tok_info = tokenizer.tokenize_text(chk["texto"])
            tokenized_chunks.append({
                "posicao": chk["posicao_chunk"],
                "tipo": chk["tipo_secao"],
                "texto": chk["texto"],
                "token_stats": tok_info,
                "metadados": chk["metadados"]
            })
            texts_to_embed.append(chk["texto"])
        tokenization_time_ms = round((time.time() - t2) * 1000, 2)

        # 5. FastEmbed
        t3 = time.time()
        vectors = embedder.embed_texts(texts_to_embed)
        embedding_time_ms = round((time.time() - t3) * 1000, 2)

        # 6. Gravação no Supabase PGVECTOR
        t4 = time.time()
        db_chunks = []
        for chk, vec, tok_item in zip(raw_chunks, vectors, tokenized_chunks):
            # Guardamos os dados estruturados e estatísticas nos metadados jsonb
            enhanced_metadata = {
                **chk["metadados"],
                "nome_arquivo": pdf_filename,
                "total_tokens": tok_item["token_stats"]["total_tokens"],
                "secao": chk["tipo_secao"]
            }
            db_chunks.append({
                "documento_id": doc_id,
                "texto": chk["texto"],
                "vetor": vec,
                "metadados": enhanced_metadata
            })
        saved_chunks = db.insert_chunks(db_chunks)
        if len(saved_chunks) != len(db_chunks):
            raise RuntimeError(
                f"Foram gravados {len(saved_chunks)} de {len(db_chunks)} chunks no Supabase."
            )
        db_time_ms = round((time.time() - t4) * 1000, 2)

        # =========================================================================
        # [CÓDIGO ADICIONADO - CONDIÇÃO 2: ARQUIVAMENTO APÓS GRAVAÇÃO NO SUPABASE]
        # Se veio via Webhook: NÃO executa o mover da pasta raw (pois o arquivo não
        # passou por lá). Em vez disso, grava o binário diretamente na pasta gold.
        # Se veio da pasta raw: executa o mover tradicional da pasta raw para gold.
        # =========================================================================
        if from_webhook or file_bytes is not None:
            archived_path = RAGService.save_webhook_document_to_gold(pdf_filename, file_bytes)
        else:
            archived_path = RAGService.archive_ingested_document(pdf_filename)

        total_time_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "documento_id": doc_id,
            "nome_arquivo": pdf_filename,
            "origem": "webhook" if (from_webhook or file_bytes is not None) else "raw",
            "arquivo_gold": os.path.basename(archived_path),
            "status": "CONCLUIDO",
            "dados_estruturados": parsed,
            "total_chunks": len(raw_chunks),
            "chunks_detalhes": [
                {
                    "posicao": tc["posicao"],
                    "tipo": tc["tipo"],
                    "texto": tc["texto"],
                    "tokens": tc["token_stats"]["total_tokens"],
                    "amostra_tokens": tc["token_stats"]["preview_tokens"][:10],
                    "vetor_preview": embedder.get_embedding_preview(v)
                }
                for tc, v in zip(tokenized_chunks, vectors)
            ],
            "metricas_tempo": {
                "extracao_ocr_ms": extraction_time_ms,
                "chunking_ms": chunking_time_ms,
                "tokenizacao_ms": tokenization_time_ms,
                "embedding_ms": embedding_time_ms,
                "pgvector_salvar_ms": db_time_ms,
                "tempo_total_ms": total_time_ms
            }
        }

    # =========================================================================
    # [CÓDIGO ADICIONADO]: SALVAR DIRETAMENTE NA PASTA GOLD PARA ARQUIVOS DE WEBHOOK
    # =========================================================================
    @staticmethod
    def save_webhook_document_to_gold(pdf_filename: str, file_bytes: bytes) -> str:
        """
        Salva o arquivo binário recebido via Webhook diretamente na pasta gold,
        sem passar nem mover da pasta raw, gerando sufixo único (UUID) caso já exista.
        """
        clean_filename = os.path.basename(pdf_filename)
        if not clean_filename.lower().endswith(".pdf"):
            raise ValueError("O arquivo recebido via webhook deve ser um PDF.")

        os.makedirs(settings.GOLD_DIR, exist_ok=True)
        destination_path = os.path.join(settings.GOLD_DIR, clean_filename)
        while os.path.exists(destination_path):
            filename, extension = os.path.splitext(clean_filename)
            unique_suffix = uuid.uuid4().hex
            destination_path = os.path.join(
                settings.GOLD_DIR,
                f"{filename}_{unique_suffix}{extension}",
            )

        with open(destination_path, "wb") as f:
            f.write(file_bytes)

        return destination_path

    # =========================================================================
    # [CÓDIGO ADICIONADO]: MÉTODO DE CONVENIÊNCIA PARA CHAMADAS DE WEBHOOK
    # =========================================================================
    @staticmethod
    def process_and_ingest_from_webhook(pdf_filename: str, file_bytes: bytes) -> Dict[str, Any]:
        """
        Executa a ingestão completa de um documento PDF recebido via Webhook (em memória),
        persistindo os dados no Supabase e salvando o binário diretamente na pasta gold,
        sem precisar ir para a pasta raw ou tentar movê-lo de lá.
        """
        return RAGService.process_and_ingest_document(
            pdf_filename=pdf_filename,
            file_bytes=file_bytes,
            from_webhook=True
        )

    @staticmethod
    def archive_ingested_document(pdf_filename: str) -> str:
        """Move um PDF já indexado de raw para gold, sem sobrescrever arquivos."""
        if os.path.basename(pdf_filename) != pdf_filename or not pdf_filename.lower().endswith(".pdf"):
            raise ValueError("Informe apenas o nome de um arquivo PDF da pasta raw.")

        source_path = os.path.join(settings.DOCS_DIR, pdf_filename)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Arquivo não encontrado na pasta raw: {pdf_filename}")

        os.makedirs(settings.GOLD_DIR, exist_ok=True)
        destination_path = os.path.join(settings.GOLD_DIR, pdf_filename)
        while os.path.exists(destination_path):
            filename, extension = os.path.splitext(pdf_filename)
            unique_suffix = uuid.uuid4().hex
            destination_path = os.path.join(
                settings.GOLD_DIR,
                f"{filename}_{unique_suffix}{extension}",
            )

        return shutil.move(source_path, destination_path)

    @staticmethod
    def query_rag(question: str, top_k: int = 5, min_threshold: float = 0.35) -> Dict[str, Any]:
        """Executa a consulta RAG com passos didáticos e regra estrita de ausência de dados"""
        start_time = time.time()
        pipeline_steps = []

        # Etapa A: Pergunta e Tokenização
        t0 = time.time()
        query_token_info = tokenizer.tokenize_text(question)
        pipeline_steps.append({
            "etapa": "TOKENIZACAO",
            "descricao": "Tokenização da pergunta do usuário com tiktoken (cl100k_base)",
            "detalhes": query_token_info,
            "tempo_ms": round((time.time() - t0) * 1000, 2)
        })

        # Etapa B: Embedding com FastEmbed
        t1 = time.time()
        query_vector = embedder.embed_query(question)
        emb_preview = embedder.get_embedding_preview(query_vector)
        pipeline_steps.append({
            "etapa": "EMBEDDING",
            "descricao": f"Geração do vetor numérico de {embedder.dimension} dimensões via FastEmbed ({embedder.model_name})",
            "detalhes": emb_preview,
            "tempo_ms": round((time.time() - t1) * 1000, 2)
        })

        # Etapa C: Retrieval no Supabase PGVECTOR
        t2 = time.time()
        # Buscamos até top_k candidatos com match_threshold relaxado no RPC, depois filtramos
        raw_candidates = db.match_chunks(query_vector, match_count=top_k, match_threshold=0.0)
        retrieval_time_ms = round((time.time() - t2) * 1000, 2)

        # Formatação dos candidatos recuperados
        candidatos = []
        for idx, c in enumerate(raw_candidates, 1):
            score = float(c.get("similarity", 0.0))
            metadata = c.get("metadados") or {}
            candidatos.append({
                "ranking": idx,
                "id": c.get("id"),
                "documento_id": c.get("documento_id"),
                "nome_arquivo": metadata.get("nome_arquivo", "Nota Fiscal"),
                "numero_nota": metadata.get("numero_nota", "N/A"),
                "score_similaridade": round(score, 4),
                "tipo_secao": metadata.get("tipo", metadata.get("secao", "conteudo")),
                "trecho": c.get("texto", ""),
                "metadados": metadata
            })

        pipeline_steps.append({
            "etapa": "RETRIEVAL",
            "descricao": f"Busca vetorial por similaridade de cosseno via RPC PGVECTOR (top_k={top_k})",
            "total_recuperados": len(candidatos),
            "top_k_solicitado": top_k,
            "tempo_ms": retrieval_time_ms
        })

        # ETAPA 16 — AUSÊNCIA DE DADOS
        # Se não houver candidatos ou o melhor score estiver abaixo do threshold mínimo de relevância:
        candidatos_relevantes = [c for c in candidatos if c["score_similaridade"] >= min_threshold]

        if not candidatos_relevantes:
            return {
                "pergunta": question,
                "status": "SEM_DADOS",
                "resposta": "Sem dados disponiveis na base",
                "candidatos": candidatos,
                "candidatos_relevantes": [],
                "prompt_aumentado": None,
                "pipeline_steps": pipeline_steps,
                "llm_executado": False,
                "motivo": "Nenhum candidato relevante recuperado na base vetorial para a pergunta",
                "tempo_total_ms": round((time.time() - start_time) * 1000, 2)
            }

        # Etapa D: Construção do Prompt Aumentado (ETAPA 17)
        t3 = time.time()
        contexto_textos = []
        for c in candidatos_relevantes:
            doc_ref = f"[Candidato #{c['ranking']} - NF nº {c['numero_nota']} - Similaridade: {c['score_similaridade']}]"
            contexto_textos.append(f"{doc_ref}\n{c['trecho']}")
        contexto_consolidado = "\n\n---\n\n".join(contexto_textos)

        instrucoes_sistema = (
            "Você é um assistente RAG especializado em conferência e análise de Notas Fiscais.\n"
            "DIRETRIZES FUNDAMENTAIS:\n"
            "1. Responda à pergunta do usuário de forma clara, objetiva e precisa, baseando-se EXCLUSIVAMENTE no contexto das notas fiscais fornecido abaixo.\n"
            "2. Não invente valores, produtos, datas ou fornecedores.\n"
            "3. Se a informação solicitada pelo usuário NÃO estiver presente no contexto fornecido, responda exatamente: 'Sem dados disponiveis na base'.\n"
            "4. Indique as notas fiscais correspondentes (ex: 'De acordo com a Nota Fiscal nº XXX...')."
        )

        prompt_aumentado = (
            f"=== INSTRUÇÕES DO SISTEMA ===\n{instrucoes_sistema}\n\n"
            f"=== CONTEXTO RECUPERADO DAS NOTAS FISCAIS ===\n{contexto_consolidado}\n\n"
            f"=== PERGUNTA DO USUÁRIO ===\n{question}"
        )

        pipeline_steps.append({
            "etapa": "PROMPT_AUMENTADO",
            "descricao": "Montagem do prompt contextualizado unindo Instruções + Contexto Top-K + Pergunta",
            "tamanho_contexto_chars": len(contexto_consolidado),
            "tempo_ms": round((time.time() - t3) * 1000, 2)
        })

        # Etapa E: Chamada ao LLM via OpenRouter (ETAPA 18)
        t4 = time.time()
        resposta_llm = RAGService._call_openrouter(instrucoes_sistema, contexto_consolidado, question)
        llm_time_ms = round((time.time() - t4) * 1000, 2)

        pipeline_steps.append({
            "etapa": "LLM_OPENROUTER",
            "descricao": f"Geração da resposta final via OpenRouter (modelo: {resposta_llm['modelo']})",
            "tempo_ms": llm_time_ms
        })

        return {
            "pergunta": question,
            "status": "SUCESSO",
            "resposta": resposta_llm["texto"],
            "modelo_usado": resposta_llm["modelo"],
            "uso_tokens": resposta_llm["uso_tokens"],
            "candidatos": candidatos,
            "candidatos_relevantes": candidatos_relevantes,
            "prompt_aumentado": prompt_aumentado,
            "contexto_utilizado": contexto_consolidado,
            "pipeline_steps": pipeline_steps,
            "llm_executado": True,
            "tempo_total_ms": round((time.time() - start_time) * 1000, 2)
        }

    @staticmethod
    def _call_openrouter(system_prompt: str, context: str, user_question: str) -> Dict[str, Any]:
        """Comunicação segura com a API do OpenRouter"""
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY não configurada no .env")

        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "RAG Didatico Teophilo"
        }

        user_content = (
            f"Contexto Recuperado:\n{context}\n\n"
            f"Pergunta: {user_question}"
        )

        payload = {
            "model": settings.OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.1
        }

        with httpx.Client(timeout=40.0) as client:
            resp = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return {
                        "texto": (choices[0].get("message", {}).get("content") or "").strip(),
                        "modelo": data.get("model", settings.OPENROUTER_MODEL),
                        "uso_tokens": {key: data.get("usage", {}).get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
                    }
                raise ValueError("OpenRouter não retornou uma resposta.")
            else:
                raise ValueError(f"Erro na chamada OpenRouter (status {resp.status_code})")

rag_service = RAGService()
