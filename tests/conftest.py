import os
import json
import sys
import types
import pytest
from typing import AsyncGenerator
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

# Set environment variables BEFORE importing app
os.environ.setdefault("LIVEKIT_URL", "wss://test.livekit.cloud")
os.environ.setdefault("LIVEKIT_API_KEY", "test-api-key")
os.environ.setdefault("LIVEKIT_API_SECRET", "test-api-secret-that-is-long-enough")
os.environ.setdefault("APP_ENV", "development")

if "livekit" not in sys.modules:
    livekit_module = types.ModuleType("livekit")
    api_module = types.ModuleType("livekit.api")
    protocol_module = types.ModuleType("livekit.protocol")
    room_module = types.ModuleType("livekit.protocol.room")
    agent_dispatch_module = types.ModuleType("livekit.protocol.agent_dispatch")

    class _VideoGrants:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _AccessToken:
        def __init__(self, api_key, api_secret, ttl=None):
            self.api_key = api_key
            self.api_secret = api_secret
            self.ttl = ttl
            self.identity = None
            self.name = None
            self.grants = None
            self.room_config = None

        def with_identity(self, identity):
            self.identity = identity
            return self

        def with_name(self, name):
            self.name = name
            return self

        def with_ttl(self, ttl):
            self.ttl = ttl
            return self

        def with_grants(self, grants):
            self.grants = grants
            return self

        def with_room_config(self, room_config):
            self.room_config = room_config
            return self

        def to_jwt(self):
            return "test-livekit-token"

    class _RoomConfiguration:
        def __init__(self, agents=None):
            self.agents = agents or []

    class _RoomAgentDispatch:
        def __init__(self, agent_name="", metadata=""):
            self.agent_name = agent_name
            self.metadata = metadata

    api_module.AccessToken = _AccessToken
    api_module.VideoGrants = _VideoGrants
    room_module.RoomConfiguration = _RoomConfiguration
    agent_dispatch_module.RoomAgentDispatch = _RoomAgentDispatch

    livekit_module.api = api_module
    livekit_module.protocol = protocol_module
    protocol_module.room = room_module
    protocol_module.agent_dispatch = agent_dispatch_module

    sys.modules["livekit"] = livekit_module
    sys.modules["livekit.api"] = api_module
    sys.modules["livekit.protocol"] = protocol_module
    sys.modules["livekit.protocol.room"] = room_module
    sys.modules["livekit.protocol.agent_dispatch"] = agent_dispatch_module

if "supabase" not in sys.modules:
    supabase_module = types.ModuleType("supabase")

    class _FakeResult:
        def __init__(self, data):
            self.data = data

    class _FakeTable:
        def __init__(self, db, table_name):
            self.db = db
            self.table_name = table_name
            self.rows = list(db.setdefault(table_name, []))
            self.pending_insert = None
            self.pending_update = None
            self.delete_mode = False
            self.filters = []
            self.range_start = None
            self.range_end = None

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, column, value):
            self.filters.append(("eq", column, value))
            return self

        def gte(self, column, value):
            self.filters.append(("gte", column, value))
            return self

        def lte(self, column, value):
            self.filters.append(("lte", column, value))
            return self

        def in_(self, column, values):
            self.filters.append(("in", column, values))
            return self

        def order(self, *_args, **_kwargs):
            return self

        def range(self, start, end):
            self.range_start = start
            self.range_end = end
            return self

        def limit(self, count):
            self.range_start = 0
            self.range_end = count - 1
            return self

        def insert(self, payload):
            self.pending_insert = payload
            return self

        def update(self, payload):
            self.pending_update = payload
            return self

        def delete(self):
            self.delete_mode = True
            return self

        def _apply_filters(self, rows):
            filtered = list(rows)
            for op, column, value in self.filters:
                if op == "eq":
                    filtered = [row for row in filtered if row.get(column) == value]
                elif op == "gte":
                    filtered = [row for row in filtered if row.get(column) >= value]
                elif op == "lte":
                    filtered = [row for row in filtered if row.get(column) <= value]
                elif op == "in":
                    filtered = [row for row in filtered if row.get(column) in value]
            return filtered

        def execute(self):
            table_rows = self.db.setdefault(self.table_name, [])

            if self.pending_insert is not None:
                payload = self.pending_insert
                if isinstance(payload, list):
                    table_rows.extend(payload)
                    return _FakeResult(payload)
                table_rows.append(payload)
                return _FakeResult([payload])

            filtered = self._apply_filters(table_rows)

            if self.pending_update is not None:
                updated = []
                for row in filtered:
                    row.update(self.pending_update)
                    updated.append(row)
                return _FakeResult(updated)

            if self.delete_mode:
                for row in filtered:
                    if row in table_rows:
                        table_rows.remove(row)
                return _FakeResult(filtered)

            if self.range_start is not None and self.range_end is not None:
                filtered = filtered[self.range_start:self.range_end + 1]

            return _FakeResult(filtered)

    class _FakeSupabaseClient:
        def __init__(self):
            self.db = {
                "user_credits": [
                    {
                        "user_id": "test-user-id",
                        "balance": 500000,
                        "lifetime_purchased": 500000,
                        "lifetime_used": 0,
                        "updated_at": "2026-03-20T00:00:00+00:00",
                    }
                ],
                "credit_transactions": [],
                "credit_usage_logs": [],
                "provider_usage_imports": [],
                "provider_usage_lines": [],
                "provider_reconciliation_runs": [],
                "provider_reconciliation_findings": [],
            }

        def table(self, table_name):
            return _FakeTable(self.db, table_name)

        def rpc(self, name, params):
            if name == "add_credits":
                user_id = params["p_user_id"]
                amount = params["p_amount"]
                row = next((row for row in self.db["user_credits"] if row["user_id"] == user_id), None)
                if row is None:
                    row = {
                        "user_id": user_id,
                        "balance": 0,
                        "lifetime_purchased": 0,
                        "lifetime_used": 0,
                        "updated_at": "2026-03-20T00:00:00+00:00",
                    }
                    self.db["user_credits"].append(row)
                row["balance"] += amount
                row["lifetime_purchased"] += amount
                return _FakeResult(row["balance"])
            if name == "deduct_credits":
                user_id = params["p_user_id"]
                amount = params["p_amount"]
                row = next((row for row in self.db["user_credits"] if row["user_id"] == user_id), None)
                if row is None or row["balance"] < amount:
                    raise Exception("Insufficient credits")
                row["balance"] -= amount
                row["lifetime_used"] += amount
                return _FakeResult(row["balance"])
            return _FakeResult(None)

    _fake_supabase_client = _FakeSupabaseClient()

    def _create_client(*_args, **_kwargs):
        return _fake_supabase_client

    supabase_module.create_client = _create_client
    supabase_module.Client = _FakeSupabaseClient
    sys.modules["supabase"] = supabase_module

if "stripe" not in sys.modules:
    stripe_module = types.ModuleType("stripe")

    class _SignatureVerificationError(Exception):
        pass

    class _FakeStripeSession:
        def __init__(self):
            self.id = "cs_test_123"
            self.url = "https://example.com/checkout"

    class _CheckoutSession:
        @staticmethod
        def create(**_kwargs):
            return _FakeStripeSession()

    class _Checkout:
        Session = _CheckoutSession

    class _Webhook:
        @staticmethod
        def construct_event(payload, _sig_header, _secret):
            return json.loads(payload.decode("utf-8")) if payload else {}

    class _StripeErrorNamespace:
        SignatureVerificationError = _SignatureVerificationError

    stripe_module.api_key = None
    stripe_module.checkout = _Checkout
    stripe_module.Webhook = _Webhook
    stripe_module.error = _StripeErrorNamespace
    sys.modules["stripe"] = stripe_module

if "zep_cloud" not in sys.modules:
    zep_cloud_module = types.ModuleType("zep_cloud")
    zep_cloud_client_module = types.ModuleType("zep_cloud.client")

    class _AsyncZep:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    zep_cloud_client_module.AsyncZep = _AsyncZep
    zep_cloud_module.client = zep_cloud_client_module
    sys.modules["zep_cloud"] = zep_cloud_module
    sys.modules["zep_cloud.client"] = zep_cloud_client_module

from src.main import app
from src.core.security import AuthUser
from src.api.deps import get_current_user

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client with unauthenticated access by default."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client

@pytest.fixture
def mock_auth_user() -> AuthUser:
    """Create a mock authenticated user."""
    return AuthUser({
        "sub": "test-user-id",
        "email": "test@example.com",
        "role": "authenticated",
        "aud": "authenticated"
    })

@pytest.fixture
async def auth_client(mock_auth_user) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client that is authenticated."""
    # Override the auth dependency
    app.dependency_overrides[get_current_user] = lambda: mock_auth_user
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client
    
    # Clean up override
    app.dependency_overrides.clear()
