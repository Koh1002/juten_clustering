"""
LLMプロバイダ抽象化モジュール
OpenAI / Gemini / Claude 対応
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

    @abstractmethod
    def list_models(self) -> List[Dict]:
        """利用可能モデル一覧を取得"""
        pass

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

    # 埋め込みモデル優先順位
    EMBEDDING_PRIORITY = [
        'text-embedding-3-large',
        'text-embedding-3-small',
        'text-embedding-ada-002',
    ]

    # 生成モデル優先順位
    GENERATION_PRIORITY = [
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-4-turbo',
        'gpt-4',
        'gpt-3.5-turbo',
    ]

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai パッケージがインストールされていません")
        return self._client

    def validate_key(self) -> bool:
        try:
            client = self._get_client()
            # モデル一覧取得で検証
            client.models.list()
            return True
        except Exception as e:
            self.validation_error = str(e)
            logger.error(f"OpenAI validation error: {e}")
            return False

    def list_models(self) -> List[Dict]:
        try:
            client = self._get_client()
            models = client.models.list()
            self.available_models = []

            for model in models.data:
                model_info = {
                    'id': model.id,
                    'created': model.created,
                    'owned_by': model.owned_by,
                }
                self.available_models.append(model_info)

            # 作成日でソート
            self.available_models.sort(key=lambda x: x.get('created', 0), reverse=True)
            self.last_updated = datetime.now()
            return self.available_models

        except Exception as e:
            logger.error(f"OpenAI list_models error: {e}")
            return []

    def pick_latest(self, task_type: str) -> Optional[str]:
        if not self.available_models:
            self.list_models()

        model_ids = [m['id'] for m in self.available_models]

        if task_type == 'embedding':
            for pref in self.EMBEDDING_PRIORITY:
                if pref in model_ids:
                    self.embedding_model = pref
                    return pref
            # フォールバック：embedding含むモデルを探索
            for mid in model_ids:
                if 'embedding' in mid.lower():
                    self.embedding_model = mid
                    return mid

        elif task_type == 'generation':
            for pref in self.GENERATION_PRIORITY:
                if pref in model_ids:
                    self.generation_model = pref
                    return pref
            # フォールバック：gptモデルを探索
            for mid in model_ids:
                if 'gpt' in mid.lower() and 'instruct' not in mid.lower():
                    self.generation_model = mid
                    return mid

        return None

    def embed(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        if not self.embedding_model:
            self.pick_latest('embedding')

        if not self.embedding_model:
            raise ValueError("埋め込みモデルが見つかりません")

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
        if not self.generation_model:
            self.pick_latest('generation')

        if not self.generation_model:
            raise ValueError("生成モデルが見つかりません")

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

    # 埋め込みモデル優先順位
    EMBEDDING_PRIORITY = [
        'text-embedding-004',
        'embedding-001',
    ]

    # 生成モデル優先順位
    GENERATION_PRIORITY = [
        'gemini-2.0-flash-exp',
        'gemini-1.5-pro-latest',
        'gemini-1.5-pro',
        'gemini-1.5-flash-latest',
        'gemini-1.5-flash',
        'gemini-pro',
    ]

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._configured = False

    def _configure(self):
        if not self._configured:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._configured = True
            except ImportError:
                raise ImportError("google-generativeai パッケージがインストールされていません")

    def validate_key(self) -> bool:
        try:
            self._configure()
            import google.generativeai as genai
            # モデル一覧取得で検証
            list(genai.list_models())
            return True
        except Exception as e:
            self.validation_error = str(e)
            logger.error(f"Gemini validation error: {e}")
            return False

    def list_models(self) -> List[Dict]:
        try:
            self._configure()
            import google.generativeai as genai
            models = list(genai.list_models())
            self.available_models = []

            for model in models:
                model_info = {
                    'id': model.name.replace('models/', ''),
                    'display_name': getattr(model, 'display_name', model.name),
                    'supported_methods': list(model.supported_generation_methods) if hasattr(model, 'supported_generation_methods') else [],
                }
                self.available_models.append(model_info)

            self.last_updated = datetime.now()
            return self.available_models

        except Exception as e:
            logger.error(f"Gemini list_models error: {e}")
            return []

    def pick_latest(self, task_type: str) -> Optional[str]:
        if not self.available_models:
            self.list_models()

        model_ids = [m['id'] for m in self.available_models]

        if task_type == 'embedding':
            for pref in self.EMBEDDING_PRIORITY:
                if pref in model_ids:
                    self.embedding_model = pref
                    return pref
            # フォールバック
            for mid in model_ids:
                if 'embedding' in mid.lower():
                    self.embedding_model = mid
                    return mid

        elif task_type == 'generation':
            for pref in self.GENERATION_PRIORITY:
                if pref in model_ids:
                    self.generation_model = pref
                    return pref
            # フォールバック
            for mid in model_ids:
                if 'gemini' in mid.lower():
                    self.generation_model = mid
                    return mid

        return None

    def embed(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        self._configure()
        import google.generativeai as genai

        if not self.embedding_model:
            self.pick_latest('embedding')

        if not self.embedding_model:
            raise ValueError("埋め込みモデルが見つかりません")

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

        if not self.generation_model:
            self.pick_latest('generation')

        if not self.generation_model:
            raise ValueError("生成モデルが見つかりません")

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

    # 生成モデル優先順位
    GENERATION_PRIORITY = [
        'claude-opus-4-20250514',
        'claude-sonnet-4-20250514',
        'claude-3-5-sonnet-20241022',
        'claude-3-5-sonnet-20240620',
        'claude-3-opus-20240229',
        'claude-3-sonnet-20240229',
        'claude-3-haiku-20240307',
    ]

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic パッケージがインストールされていません")
        return self._client

    def validate_key(self) -> bool:
        try:
            client = self._get_client()
            # 最小リクエストで検証
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception as e:
            self.validation_error = str(e)
            logger.error(f"Claude validation error: {e}")
            return False

    def list_models(self) -> List[Dict]:
        # Claudeはモデル一覧APIがないため、既知のモデルを返す
        self.available_models = [
            {'id': m, 'type': 'generation'} for m in self.GENERATION_PRIORITY
        ]
        self.last_updated = datetime.now()
        return self.available_models

    def pick_latest(self, task_type: str) -> Optional[str]:
        if task_type == 'embedding':
            # Claudeは埋め込みをサポートしない
            return None

        elif task_type == 'generation':
            # 優先順位の最初から利用可能なものを選択
            for model in self.GENERATION_PRIORITY:
                try:
                    # 実際に利用可能か確認
                    client = self._get_client()
                    response = client.messages.create(
                        model=model,
                        max_tokens=10,
                        messages=[{"role": "user", "content": "test"}]
                    )
                    self.generation_model = model
                    return model
                except Exception:
                    continue

            # フォールバック
            self.generation_model = self.GENERATION_PRIORITY[-1]
            return self.generation_model

        return None

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "Claude は埋め込みAPIを提供していません。"
            "OpenAI または Gemini のAPIキーを使用してください。"
        )

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        if not self.generation_model:
            self.pick_latest('generation')

        if not self.generation_model:
            raise ValueError("生成モデルが見つかりません")

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

    # OpenAI: sk-で始まる
    if key.startswith('sk-'):
        return 'openai'

    # Anthropic: sk-ant-で始まる
    if key.startswith('sk-ant-'):
        return 'claude'

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

    # 推定順序を決定
    if hint == 'openai':
        providers_to_try = [OpenAIProvider, GeminiProvider, ClaudeProvider]
    elif hint == 'claude':
        providers_to_try = [ClaudeProvider, OpenAIProvider, GeminiProvider]
    elif hint == 'gemini':
        providers_to_try = [GeminiProvider, OpenAIProvider, ClaudeProvider]
    else:
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
