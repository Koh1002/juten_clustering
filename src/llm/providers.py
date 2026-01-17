"""
LLMプロバイダ抽象化モジュール
OpenAI / Gemini / Claude 対応

モデル固定指定版（2026年1月）:
- OpenAI: gpt-5.2 / text-embedding-3-large
- Gemini: gemini-3-pro-preview / gemini-embedding-001
- Claude: claude-opus-4-5 (埋め込み非対応)
"""

import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """LLMプロバイダ抽象基底クラス"""

    name: str = "base"
    supports_embedding: bool = False

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embedding_model: Optional[str] = None
        self.generation_model: Optional[str] = None
        self.available_models: List[Dict] = []
        self.validation_error: Optional[str] = None
        self.last_updated: Optional[datetime] = None

    @abstractmethod
    def validate_key(self) -> bool:
        """APIキーの検証"""
        pass

    def list_models(self) -> List[Dict]:
        """利用可能モデル一覧を取得（固定リストを返す）"""
        self.last_updated = datetime.now()
        return self.available_models

    @abstractmethod
    def pick_latest(self, task_type: str) -> Optional[str]:
        """
        最新モデルを選択
        task_type: 'embedding' or 'generation'
        """
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """テキストの埋め込みベクトルを生成"""
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """テキスト生成"""
        pass

    def get_model_info(self) -> Dict:
        """現在のモデル情報を取得"""
        return {
            'provider': self.name,
            'embedding_model': self.embedding_model,
            'generation_model': self.generation_model,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'top_models': self.available_models[:5] if self.available_models else []
        }


class OpenAIProvider(LLMProvider):
    """OpenAI プロバイダ"""

    name = "OpenAI"
    supports_embedding = True

    # 固定モデル（2026年1月）
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
    DEFAULT_GENERATION_MODEL = "gpt-5.2"

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = None
        # 固定モデルを設定
        self.embedding_model = self.DEFAULT_EMBEDDING_MODEL
        self.generation_model = self.DEFAULT_GENERATION_MODEL
        self.available_models = [
            {'id': self.DEFAULT_GENERATION_MODEL, 'type': 'generation'},
            {'id': self.DEFAULT_EMBEDDING_MODEL, 'type': 'embedding'},
        ]

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai パッケージがインストールされていません")
        return self._client

    def validate_key(self) -> bool:
        """軽量な検証（埋め込みAPIで1トークンだけ試行）"""
        try:
            client = self._get_client()
            # 最小の埋め込みリクエストで検証
            client.embeddings.create(
                model=self.DEFAULT_EMBEDDING_MODEL,
                input="test"
            )
            self.last_updated = datetime.now()
            return True
        except Exception as e:
            self.validation_error = str(e)
            logger.error(f"OpenAI validation error: {e}")
            return False

    def pick_latest(self, task_type: str) -> Optional[str]:
        """固定モデルを返す"""
        self.last_updated = datetime.now()
        if task_type == 'embedding':
            return self.embedding_model
        elif task_type == 'generation':
            return self.generation_model
        return None

    def embed(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        client = self._get_client()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # 空文字列を置換
            batch = [t if t.strip() else " " for t in batch]

            response = client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            # レート制限対策
            if i + batch_size < len(texts):
                time.sleep(0.1)

        return all_embeddings

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.generation_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content


class GeminiProvider(LLMProvider):
    """Google Gemini プロバイダ"""

    name = "Gemini"
    supports_embedding = True

    # 固定モデル（2026年1月）
    DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
    DEFAULT_GENERATION_MODEL = "gemini-3-pro-preview"

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._configured = False
        # 固定モデルを設定
        self.embedding_model = self.DEFAULT_EMBEDDING_MODEL
        self.generation_model = self.DEFAULT_GENERATION_MODEL
        self.available_models = [
            {'id': self.DEFAULT_GENERATION_MODEL, 'type': 'generation'},
            {'id': self.DEFAULT_EMBEDDING_MODEL, 'type': 'embedding'},
        ]

    def _configure(self):
        if not self._configured:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._configured = True
            except ImportError:
                raise ImportError("google-generativeai パッケージがインストールされていません")

    def validate_key(self) -> bool:
        """軽量な検証（埋め込みAPIで1トークンだけ試行）"""
        try:
            self._configure()
            import google.generativeai as genai
            # 最小の埋め込みリクエストで検証
            genai.embed_content(
                model=f"models/{self.DEFAULT_EMBEDDING_MODEL}",
                content="test",
                task_type="clustering"
            )
            self.last_updated = datetime.now()
            return True
        except Exception as e:
            self.validation_error = str(e)
            logger.error(f"Gemini validation error: {e}")
            return False

    def pick_latest(self, task_type: str) -> Optional[str]:
        """固定モデルを返す"""
        self.last_updated = datetime.now()
        if task_type == 'embedding':
            return self.embedding_model
        elif task_type == 'generation':
            return self.generation_model
        return None

    def embed(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        self._configure()
        import google.generativeai as genai

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t if t.strip() else " " for t in batch]

            result = genai.embed_content(
                model=f"models/{self.embedding_model}",
                content=batch,
                task_type="clustering"
            )

            if 'embedding' in result:
                # 単一テキストの場合
                all_embeddings.append(result['embedding'])
            elif 'embeddings' in result:
                # バッチの場合
                all_embeddings.extend(result['embeddings'])

            if i + batch_size < len(texts):
                time.sleep(0.1)

        return all_embeddings

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        self._configure()
        import google.generativeai as genai

        model = genai.GenerativeModel(self.generation_model)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7
            )
        )
        return response.text


class ClaudeProvider(LLMProvider):
    """Anthropic Claude プロバイダ"""

    name = "Claude"
    supports_embedding = False  # Claudeは埋め込みAPIを提供していない

    # 固定モデル（2026年1月）
    DEFAULT_GENERATION_MODEL = "claude-opus-4-5"

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = None
        # 固定モデルを設定
        self.generation_model = self.DEFAULT_GENERATION_MODEL
        self.available_models = [
            {'id': self.DEFAULT_GENERATION_MODEL, 'type': 'generation'},
        ]

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic パッケージがインストールされていません")
        return self._client

    def validate_key(self) -> bool:
        """軽量な検証（最小リクエスト）"""
        try:
            client = self._get_client()
            # 最小リクエストで検証
            response = client.messages.create(
                model=self.DEFAULT_GENERATION_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            self.last_updated = datetime.now()
            return True
        except Exception as e:
            self.validation_error = str(e)
            logger.error(f"Claude validation error: {e}")
            return False

    def pick_latest(self, task_type: str) -> Optional[str]:
        """固定モデルを返す"""
        self.last_updated = datetime.now()
        if task_type == 'embedding':
            # Claudeは埋め込みをサポートしない
            return None
        elif task_type == 'generation':
            return self.generation_model
        return None

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "Claude は埋め込みAPIを提供していません。"
            "OpenAI または Gemini のAPIキーを使用してください。"
        )

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self.generation_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


def detect_provider_by_key(api_key: str) -> Optional[str]:
    """
    APIキーのプレフィックスからプロバイダを推定
    """
    key = api_key.strip()

    # Anthropic: sk-ant-で始まる（OpenAIより先にチェック）
    if key.startswith('sk-ant-'):
        return 'claude'

    # OpenAI: sk-で始まる
    if key.startswith('sk-'):
        return 'openai'

    # Gemini/Google: AIzaで始まることが多い
    if key.startswith('AIza'):
        return 'gemini'

    return None


def detect_provider(api_key: str) -> Tuple[Optional[LLMProvider], str]:
    """
    APIキーからプロバイダを検出し、インスタンスを返す
    戻り値: (provider_instance, status_message)
    """
    if not api_key or not api_key.strip():
        return None, "APIキーが入力されていません"

    key = api_key.strip()

    # ヒューリスティックで推定
    hint = detect_provider_by_key(key)

    # 推定されたプロバイダのみ試行（高速化）
    if hint == 'openai':
        providers_to_try = [OpenAIProvider]
    elif hint == 'claude':
        providers_to_try = [ClaudeProvider]
    elif hint == 'gemini':
        providers_to_try = [GeminiProvider]
    else:
        # 推定できない場合は全て試行
        providers_to_try = [OpenAIProvider, GeminiProvider, ClaudeProvider]

    errors = []

    for ProviderClass in providers_to_try:
        try:
            provider = ProviderClass(key)
            if provider.validate_key():
                return provider, f"{provider.name} として認証成功"
            else:
                errors.append(f"{provider.name}: {provider.validation_error}")
        except Exception as e:
            errors.append(f"{ProviderClass.name}: {str(e)}")

    error_msg = "認証失敗:\n" + "\n".join(errors)
    return None, error_msg


def get_provider(api_key: str, provider_name: str) -> Optional[LLMProvider]:
    """
    指定されたプロバイダのインスタンスを取得
    """
    providers = {
        'openai': OpenAIProvider,
        'gemini': GeminiProvider,
        'claude': ClaudeProvider,
    }

    ProviderClass = providers.get(provider_name.lower())
    if not ProviderClass:
        return None

    return ProviderClass(api_key)
