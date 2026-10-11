import os
import time
from pathlib import Path
from dotenv import load_dotenv

import requests

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class GatewayError(Exception):
    """A problem talking to the WSO2 gateway, with a message that is safe to show to a user."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class GatewayClient:
    """Calls the forex API through the WSO2 gateway using the OAuth2 client-credentials grant."""

    def __init__(self, base_url=None, token_url=None, key=None, secret=None, verify_tls=None, timeout=10):
        self.base_url = (base_url or os.getenv("GATEWAY_BASE_URL", "https://localhost:8243/forex/1.0.0")).rstrip("/")
        self.token_url = token_url or os.getenv("TOKEN_URL", "https://localhost:9443/oauth2/token")
        self.key = key or os.getenv("WSO2_CONSUMER_KEY")
        self.secret = secret or os.getenv("WSO2_CONSUMER_SECRET")
        if verify_tls is None:
            verify_tls = os.getenv("VERIFY_TLS", "true").lower() != "false"   # set VERIFY_TLS=false only for a local self-signed WSO2
        self.verify_tls = verify_tls
        self.timeout = timeout
        self._token = None
        self._expires_at = 0.0
        if not self.verify_tls:
            requests.packages.urllib3.disable_warnings()

    def _get_token(self, force: bool = False) -> str:
        if not force and self._token and time.time() < self._expires_at - 30:
            return self._token
        if not self.key or not self.secret:
            raise GatewayError(0, "WSO2_CONSUMER_KEY and WSO2_CONSUMER_SECRET are not set.")
        try:
            r = requests.post(self.token_url, data={"grant_type": "client_credentials"},
                              auth=(self.key, self.secret), verify=self.verify_tls, timeout=self.timeout)
        except requests.exceptions.RequestException:
            raise GatewayError(0, "Cannot reach the WSO2 token endpoint. Is WSO2 running?")
        if r.status_code != 200:
            raise GatewayError(r.status_code, "WSO2 refused to issue a token. Check the consumer key and secret.")
        data = r.json()
        self._token = data["access_token"]
        self._expires_at = time.time() + int(data.get("expires_in", 3600))
        return self._token

    def _request(self, method: str, path: str, **kwargs):
        r = None
        for attempt in (1, 2):
            token = self._get_token(force=(attempt == 2))
            try:
                r = requests.request(method, self.base_url + path, headers={"Authorization": f"Bearer {token}"},
                                     verify=self.verify_tls, timeout=self.timeout, **kwargs)
            except requests.exceptions.RequestException:
                raise GatewayError(0, "Cannot reach the WSO2 gateway. Are WSO2 and the API running?")
            if r.status_code == 401 and attempt == 1:
                continue                      # the token may have expired: fetch a new one and retry once
            break
        if r.status_code == 429:
            raise GatewayError(429, "Rate limit reached for this subscription plan. Wait a minute and try again.")
        if r.status_code == 503:
            raise GatewayError(503, "The model service is not available right now.")
        if r.status_code == 404:
            raise GatewayError(404, "The gateway has no such resource. Redeploy the API in WSO2.")
        if not r.ok:
            raise GatewayError(r.status_code, f"The gateway returned an error ({r.status_code}).")
        return r.json()

    def currencies(self) -> dict:
        return self._request("GET", "/currencies")

    def model_info(self) -> dict:
        return self._request("GET", "/model-info")

    def predict(self, currency: str) -> dict:
        return self._request("POST", "/predict", json={"currency": currency})