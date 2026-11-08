"""Encryption at rest for the stored OAuth tokens (spec §4.2; 0017 §2).

Fernet (AES-128-CBC + HMAC-SHA256) with the key from
`OAUTH_TOKEN_ENCRYPTION_KEY`. A wrong or rotated key surfaces as
`TokenDecryptError`, which `tokens.py` turns into `needs_reconsent`
rather than a crash: re-consent rewrites the row with the current key.
"""

from cryptography.fernet import Fernet, InvalidToken

from app.config import Settings


class TokenDecryptError(Exception):
    pass


class TokenCipher:
    def __init__(self, key: str) -> None:
        try:
            self._fernet = Fernet(key.encode())
        except (ValueError, TypeError) as exc:
            # Never echo the key itself.
            raise ValueError("OAUTH_TOKEN_ENCRYPTION_KEY is not a valid Fernet key") from exc

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken as exc:
            raise TokenDecryptError("token decryption failed") from exc


def cipher_from_settings(settings: Settings) -> TokenCipher:
    if settings.oauth_token_encryption_key is None:
        raise ValueError("OAUTH_TOKEN_ENCRYPTION_KEY is not set")
    return TokenCipher(settings.oauth_token_encryption_key.get_secret_value())
