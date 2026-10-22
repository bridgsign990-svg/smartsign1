import os
import time
from unittest.mock import patch

import pytest
from itsdangerous import SignatureExpired

import app
import config


def _invite_token(room_id="room123"):
    return app._call_invite_serializer().dumps({"room_id": room_id})


def test_guest_invite_grants_only_its_room_and_displays_expiry():
    client = app.app.test_client()
    token = _invite_token()

    response = client.get(f"/call/invite/{token}")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/call/room123")

    page = client.get("/call/room123")
    assert page.status_code == 200
    assert b"CALL_USERNAME = \"Guest\"" in page.data
    assert b"CALL_INVITE_URL" in page.data
    assert b"Guest invitation expires after" in page.data
    assert client.get("/call/otherroom").status_code == 302


def test_guest_room_permission_expires_with_invitation_session():
    client = app.app.test_client()
    client.get(f"/call/invite/{_invite_token()}")
    with client.session_transaction() as browser_session:
        browser_session["call_guest_expires_at"] = time.time() - 1

    assert client.get("/call/room123").status_code == 302
    with client.session_transaction() as browser_session:
        assert "call_guest_room_id" not in browser_session
        assert "call_guest_invite_token" not in browser_session


def test_invalid_and_expired_guest_invites_are_rejected():
    client = app.app.test_client()
    assert client.get("/call/invite/not-a-valid-signature").status_code == 410

    class ExpiredSerializer:
        def loads(self, *args, **kwargs):
            raise SignatureExpired("expired")

    with patch.object(
        app, "_call_invite_serializer", return_value=ExpiredSerializer()
    ):
        expired = client.get("/call/invite/expired-token")
    assert expired.status_code == 410
    assert b"expired" in expired.data.lower()


def test_public_call_url_is_used_and_requires_https():
    with app.app.test_request_context("/"):
        with patch.dict(os.environ, {"BRIDGESIGN_PUBLIC_URL": "https://sign.example.org"}):
            link = app._make_call_invite("room123")
        assert link.startswith("https://sign.example.org/call/invite/")

        with patch.dict(os.environ, {"BRIDGESIGN_PUBLIC_URL": "http://sign.example.org"}):
            try:
                app._make_call_invite("room123")
            except ValueError as error:
                assert "HTTPS" in str(error)
            else:
                raise AssertionError("A non-HTTPS public call URL must be rejected.")


def test_loopback_and_non_https_invite_links_show_warning():
    assert app._call_invite_warning("http://127.0.0.1:5000/call/invite/token")
    assert app._call_invite_warning("http://192.168.1.2/call/invite/token")
    with patch.object(config, "SECRET_KEY_IS_EPHEMERAL", False):
        assert not app._call_invite_warning("https://sign.example.org/call/invite/token")


def test_ephemeral_secret_key_rejects_remote_guest_invites():
    client = app.app.test_client()
    with patch.object(config, "SECRET_KEY_IS_EPHEMERAL", True):
        token = _invite_token()
        response = client.get(
            f"/call/invite/{token}",
            environ_base={"REMOTE_ADDR": "203.0.113.10"},
        )
    assert response.status_code == 503
    assert b"private SECRET_KEY" in response.data


def test_missing_secret_generates_an_ephemeral_local_key():
    with patch.dict(os.environ, {}, clear=True):
        secret_key, is_ephemeral = config._load_secret_key()

    assert len(secret_key) >= 64
    assert is_ephemeral


@pytest.mark.parametrize(
    ("production_marker", "production_value"),
    [
        ("RENDER", "true"),
        ("RAILWAY_ENVIRONMENT", "production"),
        ("FLASK_ENV", "production"),
        ("APP_ENV", "production"),
        ("ENVIRONMENT", "production"),
    ],
)
def test_missing_secret_is_rejected_in_production(production_marker, production_value):
    with patch.dict(os.environ, {production_marker: production_value}, clear=True):
        with pytest.raises(RuntimeError, match="SECRET_KEY must be configured in production"):
            config._load_secret_key()


def test_explicit_secret_is_used_without_ephemeral_fallback():
    with patch.dict(os.environ, {"SECRET_KEY": "unit-test-only-value"}, clear=True):
        assert config._load_secret_key() == ("unit-test-only-value", False)


def test_turn_servers_are_optional_and_validated():
    with patch.dict(os.environ, {"BRIDGESIGN_TURN_ICE_SERVERS": ""}):
        assert app._call_turn_ice_servers() == []

    with patch.dict(
        os.environ,
        {
            "BRIDGESIGN_TURN_ICE_SERVERS": (
                '[{"urls":"turns:relay.example.org:5349",'
                '"username":"temporary-user","credential":"temporary-secret"}]'
            )
        },
    ):
        assert app._call_turn_ice_servers() == [{
            "urls": "turns:relay.example.org:5349",
            "username": "temporary-user",
            "credential": "temporary-secret",
        }]

    with patch.dict(
        os.environ,
        {"BRIDGESIGN_TURN_ICE_SERVERS": '[{"urls":"stun:relay.example.org"}]'},
    ):
        try:
            app._call_turn_ice_servers()
        except ValueError as error:
            assert "turn: or turns:" in str(error)
        else:
            raise AssertionError("A non-TURN server URL must be rejected.")
