#!/usr/bin/env python3
"""
Lightweight smoke tester for RAG modules.

This script injects minimal stub implementations for the langchain-related
packages used by the RAG modules in this repo so we can import and exercise
`execute_rag(question)` without installing the real dependencies or calling
external APIs.

It is intentionally conservative: it tries to mimic only the small surface
area needed by the repository's RAG modules (TextLoader, FAISS-like vector
store, simple retrievers, and chain-like objects with an `invoke` method).

Use this to quickly validate the RAG modules' interfaces and basic control
flow. It does not attempt to validate model outputs or embeddings.
"""

import sys
import types
import json
import importlib
from pathlib import Path


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class TextLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        p = Path(self.path)
        if not p.exists():
            return [Document("")]
        text = p.read_text(encoding="utf-8")
        # Return a single Document containing the whole text
        return [Document(text)]


class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, docs):
        out = []
        for d in docs:
            text = d.page_content
            if not text:
                continue
            # naive chunking by characters
            i = 0
            while i < len(text):
                chunk = text[i : i + self.chunk_size]
                out.append(Document(chunk))
                i += max(1, self.chunk_size - self.chunk_overlap)
        return out


class OpenAIEmbeddings:
    def __init__(self, *args, **kwargs):
        pass


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs or []

    @classmethod
    def from_documents(cls, docs, embeddings=None):
        return cls(docs)

    @classmethod
    def from_texts(cls, texts, embeddings=None):
        docs = [Document(t) for t in texts]
        return cls(docs)

    def as_retriever(self, search_kwargs=None):
        k = None
        if isinstance(search_kwargs, dict):
            k = search_kwargs.get("k")
        return FakeRetriever(self.docs, k=k)

    def similarity_search(self, query, k=5):
        return self.docs[:k]

    def get_relevant_documents(self, query):
        return self.docs[:5]


class FakeRetriever:
    def __init__(self, docs, k=None):
        self.docs = docs
        self.k = k or 5

    def invoke(self, q):
        return self.docs[: self.k]

    def get_relevant_documents(self, q):
        return self.docs[: self.k]


class ChatPromptTemplate:
    @classmethod
    def from_template(cls, t):
        inst = cls()
        inst.template = t
        return inst

    def __or__(self, other):
        return CombinedChain([self, other])


class ChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass

    def __or__(self, other):
        return CombinedChain([self, other])


class RunnablePassthrough:
    def __or__(self, other):
        return CombinedChain([self, other])


class RunnableLambda:
    def __init__(self, fn):
        self.fn = fn

    def __or__(self, other):
        return CombinedChain([self, other])


class StrOutputParser:
    def __or__(self, other):
        return CombinedChain([self, other])


class CombinedChain:
    def __init__(self, parts=None):
        self.parts = parts or []

    def __or__(self, other):
        if isinstance(other, CombinedChain):
            return CombinedChain(self.parts + other.parts)
        return CombinedChain(self.parts + [other])

    def invoke(self, inp):
        # For the smoke test, return a deterministic, short answer and a
        # representation of the provided input so callers can inspect it.
        try:
            if isinstance(inp, dict):
                q = inp.get("question") or inp.get("query") or str(inp)
            else:
                q = str(inp)
        except Exception:
            q = str(inp)
        return f"FAKE_ANSWER for: {q[:120]}"


class BM25Retriever:
    @classmethod
    def from_documents(cls, docs):
        inst = cls()
        inst.docs = docs
        inst.k = 5
        return inst

    def get_relevant_documents(self, q):
        return self.docs[: getattr(self, "k", 5)]


class EnsembleRetriever:
    def __init__(self, retrievers, weights=None, search_type=None):
        self.retrievers = retrievers

    def get_relevant_documents(self, q):
        out = []
        for r in self.retrievers:
            if hasattr(r, "get_relevant_documents"):
                out.extend(r.get_relevant_documents(q))
        return out[:5]


class InMemoryStore:
    def __init__(self):
        self._m = {}

    def mset(self, pairs):
        for k, v in pairs:
            self._m[k] = v

    def mget(self, keys):
        return [self._m.get(k) for k in keys]


def inject_stubs():
    """Inject lightweight stub modules into sys.modules so imports succeed."""
    # helper to make module and set attributes
    def make_mod(name, attrs=None):
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    # core stubs
    make_mod("langchain_community.document_loaders", {"TextLoader": TextLoader})
    make_mod("langchain_text_splitters", {"RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter})
    make_mod("langchain_openai", {"OpenAIEmbeddings": OpenAIEmbeddings, "ChatOpenAI": ChatOpenAI})
    make_mod("langchain_community.vectorstores", {"FAISS": FakeVectorStore})
    make_mod("langchain_core.prompts", {"ChatPromptTemplate": ChatPromptTemplate})
    make_mod("langchain_core.runnables", {"RunnablePassthrough": RunnablePassthrough, "RunnableLambda": RunnableLambda})
    make_mod("langchain_core.output_parsers", {"StrOutputParser": StrOutputParser})
    make_mod("langchain_core.documents", {"Document": Document})
    make_mod("langchain_community.retrievers", {"BM25Retriever": BM25Retriever})
    make_mod("langchain", {})
    make_mod("langchain.retrievers", {"EnsembleRetriever": EnsembleRetriever, "ParentDocumentRetriever": type("ParentDocumentRetriever", (), {"__init__": lambda self, **kw: None, "add_documents": lambda self, docs: None, "get_relevant_documents": lambda self, q: []})})
    make_mod("langchain.storage", {"InMemoryStore": InMemoryStore})
    make_mod("langchain.load", {"dumps": lambda x: "", "loads": lambda x: None})
    # also provide top-level names used sometimes
    make_mod("langchain_community", {})


def run_smoke_test():
    cwd = Path(__file__).resolve().parents[2]
    cfg_path = cwd / "config.json"
    if not cfg_path.exists():
        print("config.json not found; aborting")
        return 2

    with cfg_path.open() as fh:
        cfg = json.load(fh)

    inject_stubs()

    # ensure src on path so package imports like chatbot.rags.* resolve
    sys.path.insert(0, str(cwd))

    rags = [k for k, v in cfg.get("rags", {}).items() if v.get("enabled")]
    print(f"Found RAGs in config: {rags}")

    results = {}
    for key, info in cfg.get("rags", {}).items():
        if not info.get("enabled"):
            continue
        module_name = info.get("module")
        print(f"\n--- Testing module: {module_name} (key={key})")
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print(f"IMPORT FAILED: {e}")
            results[key] = {"import": False, "error": str(e)}
            continue

        if not hasattr(mod, "execute_rag"):
            print("MODULE MISSING execute_rag function")
            results[key] = {"import": True, "execute_rag": False}
            continue

        try:
            out = mod.execute_rag("¿Quién es Peter Pan?")
            print("execute_rag returned type:", type(out))
            # Normalized checks
            ok = isinstance(out, dict) and "answer" in out and "context" in out
            print("basic return shape ok:", ok)
            if ok:
                print("answer:", str(out.get("answer"))[:200])
                ctx = out.get("context")
                if isinstance(ctx, list):
                    print(f"context length: {len(ctx)}; sample: {str(ctx[0].page_content)[:120] if ctx else None}")
                else:
                    print(f"context (non-list): {str(ctx)[:200]}")
            results[key] = {"import": True, "execute_rag": True, "shape_ok": ok}
        except Exception as e:
            print("EXECUTION FAILED:", e)
            results[key] = {"import": True, "execute_rag": True, "error": str(e)}

    print('\nSummary:')
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())
